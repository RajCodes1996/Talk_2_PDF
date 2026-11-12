"""Extractive answers with spoken page citations: sentence scoring, TextRank summaries, type routing."""
import math
import re

import numpy as np

from src.nlp.question_classifier import extract_page, plan
from src.retrieval.index import Index, tokenize

MIN_SCORE = 0.30  # below this the answer is "not found" (tune with tools.eval_answers)
NAV = re.compile(r"\b(read|go|open|take|jump|turn|skip|start|play|back|show)\b", re.I)
EXTRA_STOP = set(tokenize("many much tell please define definition mean meaning called name stand"))
GENERIC = set(tokenize("summary summarize summarise sum up section part chapter document overview recap brief "
                       "briefly describe contents content cover covers file paper text main points key "
                       "takeaways gist highlights"))
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b|\b(january|february|march|april|may|june|july|august|september|"
                  r"october|november|december)\b", re.I)
DEFN = re.compile(r"\b(is|are|refers to|means|defined as|stands for)\b", re.I)


def cite(pages):
    p = sorted(set(pages))
    if len(p) == 1:
        return f"page {p[0]}"
    return "pages " + ", ".join(map(str, p[:-1])) + f" and {p[-1]}"


def textrank(sents, n, d=0.85, iters=30):
    """Return the indices of the n top-ranked sentences (reading order). TF-IDF cosine graph + PageRank."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    if len(sents) <= n:
        return list(range(len(sents)))
    X = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit_transform(sents)
    S = linear_kernel(X)
    np.fill_diagonal(S, 0)
    rs = S.sum(axis=1, keepdims=True)
    M = np.divide(S, rs, out=np.zeros_like(S), where=rs > 0)
    r = np.full(len(sents), 1 / len(sents))
    for _ in range(iters):
        r = (1 - d) / len(sents) + d * M.T @ r
    return sorted(np.argsort(-r, kind="stable")[:n].tolist())


class Answerer:
    def __init__(self, index, num_pages=None, qmodel=None, min_score=MIN_SCORE):
        self.idx, self.qmodel, self.min_score = index, qmodel, min_score
        self.S = []  # sentence table in reading order
        for c in index.chunks:
            for j, (t, pg) in enumerate(c["sents"]):
                prev = self.S[-1]["toks"] if j else set()  # previous sentence of the same chunk (pronoun context)
                self.S.append({"text": t, "page": pg, "chunk": c["id"], "heading": c["heading"],
                               "hid": c["heading_id"], "first": j == 0, "toks": set(tokenize(t)), "prev": prev})
        self.num_pages = num_pages or max(c["page_end"] for c in index.chunks)
        df = {}
        for s in self.S:
            for t in s["toks"]:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + len(self.S) / n) for t, n in df.items()}

    @classmethod
    def from_doc(cls, doc, **kw):
        return cls(Index.from_doc(doc), doc.get("num_pages"), **kw)

    # ---- helpers
    def _plan(self, q):
        try:
            return plan(q, self.qmodel)
        except (FileNotFoundError, OSError):  # no trained model: regex-only fallback
            pg = extract_page(q)
            return {"type": "page_lookup" if pg is not None else "other", "margin": 0.0, "page": pg, "handler": "page" if pg else "passages"}

    def _out(self, q, p, answer, sources=(), conf=0.0, found=True, action=None):
        return {"question": q, "type": p["type"], "handler": p["handler"], "page": p["page"], "answer": answer,
                "sources": list(sources), "confidence": round(conf, 3), "found": found, "action": action}

    def _rank(self, q, qtype, k=3):
        qt = set(tokenize(q)) - EXTRA_STOP
        hits = {h["chunk_id"]: h["score"] for h in self.idx.query(q, k)}
        denom = sum(self.idf.get(t, 0) for t in qt)
        if not hits or denom == 0:
            return []
        out = []
        for s in self.S:
            if s["chunk"] not in hits:
                continue
            w = len(s["text"].split())
            own = qt & s["toks"]
            cov = sum(self.idf.get(t, 0) for t in own) + 0.5 * sum(self.idf.get(t, 0) for t in (qt & s["prev"]) - own)
            sc = 0.6 * cov / denom + 0.25 * hits[s["chunk"]]
            sc += 0.05 * s["first"] - (0.2 if w < 4 else 0.1 if w > 60 else 0)
            lq = q.lower()
            if qtype == "number" and re.search(r"\d", s["text"]):
                sc += 0.3  # a numeric question needs a sentence that actually contains a number
            elif qtype == "wh_fact":
                if re.search(r"\bwhen\b|\bwhat year\b", lq) and YEAR.search(s["text"]):
                    sc += 0.15
                elif re.search(r"\bwho\b", lq) and re.search(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+", s["text"]):
                    sc += 0.15
                elif re.search(r"\bwhere\b", lq) and re.search(r"\b(in|at|near|located|based)\b", s["text"]):
                    sc += 0.15
            elif qtype == "definition" and DEFN.search(s["text"]) and qt & set(tokenize(" ".join(s["text"].split()[:8]))):
                sc += 0.15
            out.append((sc, s))
        return sorted(out, key=lambda x: -x[0])

    # ---- handlers
    def _passage(self, q, p):
        r = self._rank(q, p["type"])
        if not r or r[0][0] < self.min_score:
            return self._out(q, p, "I could not find a clear answer in this document.", found=False)
        take = [r[0]] if p["type"] != "other" else [x for x in r[:2]]
        take = sorted(take, key=lambda x: self.S.index(x[1]))
        pre = "The closest text says, on " if p["type"] == "yes_no" else "According to "
        pages = [s["page"] for _, s in take]
        ans = f"{pre}{cite(pages)}{',' if p['type'] == 'yes_no' else ':'} " + " ".join(s["text"] for _, s in take)
        src = [{"page": s["page"], "text": s["text"], "score": round(sc, 3)} for sc, s in r[:3]]
        return self._out(q, p, ans, src, min(1.0, r[0][0]))

    def _summary(self, q, p, sents, scope, n):
        pool = [s for s in sents if 6 <= len(s["text"].split()) <= 50] or sents
        if not pool:
            return self._out(q, p, "There is nothing to summarize.", found=False)
        pick = [pool[i] for i in textrank([s["text"] for s in pool], n)]
        ans = f"Summary of {scope}, from {cite([s['page'] for s in pick])}: " + " ".join(s["text"] for s in pick)
        return self._out(q, p, ans, [{"page": s["page"], "text": s["text"], "score": None} for s in pick], 1.0)

    def _page(self, q, p):
        n = p["page"]
        n = self.num_pages if n == -1 else n
        if n is None or n < 1 or n > self.num_pages:
            return self._out(q, p, f"This document has {self.num_pages} pages.", found=False)
        sents = [s for s in self.S if s["page"] == n]
        if not sents:
            return self._out(q, p, f"Page {n} has no readable text.", found=False)
        if NAV.search(q):
            return self._out(q, p, f"Going to page {n}.", conf=1.0, action={"goto_page": n})
        return self._summary(q, p, sents, f"page {n}", 2)

    def answer(self, q):
        p = self._plan(q)
        if p["type"] == "page_lookup":
            return self._page(q, p)
        if p["type"] == "summary":
            qk = set(tokenize(q)) - GENERIC - EXTRA_STOP
            best, hid = 0, None
            for s in self.S:
                o = len(qk & set(tokenize(s["heading"])))
                if s["heading"] and o > best:
                    best, hid = o, s["hid"]
            if hid is not None:
                sec = [s for s in self.S if s["hid"] == hid]
                return self._summary(q, p, sec, f"the section {sec[0]['heading']}", 3)
            return self._summary(q, p, self.S, "the document", 5)
        return self._passage(q, p)


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("question")
    ap.add_argument("--classify", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(a.source, encoding="utf-8")) if a.source.lower().endswith(".json") else \
        __import__("src.ingestion.extractor", fromlist=["x"]).extract_document(a.source)
    if a.classify:
        from pathlib import Path
        from src.nlp.block_classifier import MODEL_PATH, classify_document
        if Path(MODEL_PATH).exists():
            classify_document(doc)
    r = Answerer.from_doc(doc).answer(a.question)
    print(f"[{r['type']} conf={r['confidence']} found={r['found']} action={r['action']}]\n{r['answer']}")

