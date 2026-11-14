# task_day10.md: Extractive answers with spoken page citations (+ TextRank summaries)

Goal: `Answerer.from_doc(doc).answer(question)` returns `{type, page, answer, sources, confidence, found, action}` where `answer` is ready to speak ("According to page 4: ..."). Offline, no LLM. Routes by the Day 9 question type: definition / wh_fact / number / yes_no / other -> best sentence(s); summary -> TextRank (whole document or a named section); page_lookup -> "Going to page N" action or a 2-sentence page brief. Not found -> says so. Scope cuts (Instructions.md budget): no learned ranker, no yes/no verdicts (the closest text is read instead).

## Rules (override AGENTS.md where they conflict)
- Work in `C:\Users\netware\Desktop\vision`, project root only.
- FIRST: `python -c "import sys; print(sys.version); print(sys.executable)"` = **3.11.9**, path inside `vision\.venv`. Else STOP.
- Forbidden: uv/poetry/conda, recreating `.venv`, `pyproject.toml`/`uv.lock`, `import fitz`, new packages (no `sumy`, no nltk data), network/API/LLM calls. Never create or overwrite a PDF in `data\samples`.
- Create/edit ONLY the files named here. Use `python -m pytest`.
- Keep output short: pipe through `Select-Object -First/-Last N` as written. Full-suite runs are exempt from "narrowest test".
- Report REAL output; no claims like "all pass" without the counts. List every extra file touched under Deviations.
- Fails twice: stop, report last 20 lines plus 2 options.
- Do not tune scoring weights, add training questions or edit eval questions to improve numbers. The only allowed tuning is `MIN_SCORE` (Part E).

## Part A: Preflight (read-only)
```
python -m pytest tests/test_retrieval.py tests/test_classifier.py tests/test_questions.py -q | Select-Object -Last 3
Test-Path data\models\question_clf.joblib
git status --short | Select-Object -First 10
```
Expected: **21 passed** (9 + 3 + 9) after task_day8b.md and task_day9.md; `True`; clean tree. If the model file is missing, run `python -m tools.train_questions | Select-Object -Last 3` first (Day 9) and say so. Otherwise STOP and report.

## Part B: Edit one file, create three
1. `src/retrieval/chunker.py`: two small edits (each chunk also keeps its sentences with page numbers):
   - In `flush()`, inside the dict built for `chunks.append({...})`, add the line `"sents": [[t, pg] for t, pg in zip(cur["sents"], cur["pages"])],` right after the `"block_ids": ...` entry.
   - In the merge loop (`for c in chunks:`), add `p["sents"] += c["sents"]` as the first line after `p = out[-1]`.
   Show the diff only.
2. Create these files (exact code):

### `src/nlp/answer.py`
```python
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
```
### `tools/eval_answers.py`
```python
"""python -m tools.eval_answers [--sweep] [--exclude DOC ...] [--classify]
Uses data/eval/retrieval_eval.csv (doc,question,expected_phrase,expected_page).
heard = expected phrase is inside the spoken answer (what the user actually hears; main metric)
src3  = phrase inside any of the top-3 source sentences. wrong_abstain = answer said 'not found' but a phrase exists.
Use expected_phrase __none__ for questions the document cannot answer (correct = abstain).
Writes data/eval/answers.txt."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.nlp.answer import Answerer

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/eval/retrieval_eval.csv")
p.add_argument("--processed", default="data/processed")
p.add_argument("--samples", default="data/samples")
p.add_argument("--exclude", nargs="*", default=[])
p.add_argument("--min-score", type=float, default=0.30)
p.add_argument("--sweep", action="store_true")
p.add_argument("--classify", action="store_true")
p.add_argument("--out", default="data/eval/answers.txt")
a = p.parse_args()

rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if r["question"].strip() and r["doc"] not in a.exclude]
docs = {}
for r in rows:
    if r["doc"] in docs:
        continue
    j = Path(a.processed) / f"{r['doc']}.json"
    if j.exists():
        d = json.load(open(j, encoding="utf-8"))
    else:
        from src.ingestion.extractor import extract_document
        d = extract_document(str(Path(a.samples) / f"{r['doc']}.pdf"))
    if a.classify:
        from src.nlp.block_classifier import classify_document
        d = classify_document(d)
    docs[r["doc"]] = d


def ok(text, r):
    ph = r["expected_phrase"].strip().lower()
    return ph in text.lower() if ph and ph != "__none__" else False


def run(ms):
    ans = {k: Answerer.from_doc(v, min_score=ms) for k, v in docs.items()}
    heard = src3 = wa = 0
    words, types, bad = [], Counter(), []
    for r in rows:
        x = ans[r["doc"]].answer(r["question"])
        if r["expected_phrase"].strip().lower() == "__none__":  # unanswerable: correct = abstain
            h = s3 = not x["found"]
        else:
            h = ok(x["answer"], r)
            s3 = h or any(ok(s["text"], r) for s in x["sources"])
        heard += h
        src3 += s3
        wa += (not x["found"]) and s3 and r["expected_phrase"].strip().lower() != "__none__"
        words.append(len(x["answer"].split()))
        types[x["type"]] += 1
        if not h:
            bad.append(f"  {r['doc']} | {r['question'][:55]} | type={x['type']} found={x['found']} src3={s3}")
    n = len(rows)
    return f"min_score={ms}: heard={heard/n:.3f} src3={src3/n:.3f} wrong_abstain={wa} avg_words={sum(words)/n:.0f} types={dict(types)}", bad


lines = [f"questions={len(rows)} docs={len(docs)}"]
for ms in ([0.2, 0.3, 0.4, 0.5] if a.sweep else [a.min_score]):
    s, bad = run(ms)
    lines.append(s)
lines.append(f"not heard at min_score={ms}:")
lines += bad
print("\n".join(lines))
Path(a.out).write_text("\n".join(lines), encoding="utf-8")
```
### `tests/test_answer.py`
```python
import pytest

from src.nlp.answer import Answerer, cite, textrank
from src.nlp.question_classifier import train
from tests.test_retrieval import make_doc
from tools.make_question_data import make


@pytest.fixture(scope="module")
def ans(tmp_path_factory):
    d = tmp_path_factory.mktemp("a")
    make(d)
    return Answerer.from_doc(make_doc(), qmodel=train(d / "train.csv", d / "m.joblib"))


@pytest.mark.parametrize("q,phrase,page", [
    ("What contrast ratio is recommended?", "4.5", 3),
    ("How many cells does a braille display have?", "14 and 80", 4),
    ("What is NVDA?", "NVDA is a free", 1),
    ("Is NVDA free?", "free", 1)])
def test_answers_carry_citation(ans, q, phrase, page):
    r = ans.answer(q)
    assert r["found"] and phrase in r["answer"] and f"page {page}" in r["answer"] and r["sources"][0]["page"] == page


def test_yes_no_gives_no_verdict(ans):
    a = ans.answer("Is NVDA free?")["answer"]
    assert a.startswith("The closest text says") and not a.lower().startswith("yes")


def test_page_summary_and_abstain(ans):
    r = ans.answer("Read page 2.")
    assert r["action"] == {"goto_page": 2} and r["answer"] == "Going to page 2."
    assert "contrast" in ans.answer("What is on page 3?")["answer"].lower()
    assert ans.answer("Read page 9.")["answer"] == "This document has 4 pages."
    assert ans.answer("What is the capital of France?")["found"] is False


def test_summary_and_helpers(ans):
    r = ans.answer("Summarize this document.")
    assert r["answer"].startswith("Summary of the document") and len(r["sources"]) == 5
    sec = ans.answer("Summarize the section on braille displays.")
    assert "braille" in sec["answer"].lower() and len(sec["sources"]) <= 3
    assert cite([2]) == "page 2" and cite([3, 2, 5]) == "pages 2, 3 and 5"
    assert textrank(["a b c d e f", "a b c d e g", "x y z w v u", "a b c x y z"], 2) == sorted(textrank(["a b c d e f", "a b c d e g", "x y z w v u", "a b c x y z"], 2))
```

## Part C: Verify (agent)
```
python -m pytest tests/test_answer.py tests/test_retrieval.py -q
python -m src.nlp.answer data\samples\test_document_5pages.pdf "What contrast ratio is recommended?"
python -m src.nlp.answer data\samples\test_document_5pages.pdf "Read page 2."
python -m src.nlp.answer data\samples\test_document_5pages.pdf "Summarize this document."
python -m src.nlp.answer data\samples\sam.pdf "What is transparency diligence?"
python -m src.nlp.answer data\samples\sam.pdf "What is the capital of France?"
python -m pytest -q | Select-Object -Last 3
```
Expected: **16 passed** (7 + 9); the CLI shows `[type conf found action]` then the spoken text: contrast question cites the right page and contains `4.5`; "Read page 2." gives `Going to page 2.` with `action={'goto_page': 2}`; summary starts `Summary of the document, from pages ...`; the France question says `I could not find a clear answer in this document.` Full suite: **44** if the reader still has 2 tests (37 after Day 9, plus 7), **50** if it was restored. Report all output.

## Part D: USER step (about 5 minutes, not the agent)
In `data\eval\retrieval_eval.csv` add **5 questions the documents cannot answer** (any `doc`, real names) with `expected_phrase` exactly `__none__`. These test that the system admits it does not know instead of reading a wrong sentence aloud. Tell the agent when done.

## Part E: Evaluate (agent, after Part D)
```
python -m tools.eval_answers --exclude test_document_5pages --sweep
```
Report the full output. Rule for `MIN_SCORE` (default 0.30 in `src/nlp/answer.py`): keep 0.30 unless another tested value has a higher `heard` by at least 0.03; then change only that constant and rerun once without `--sweep`. Compare `heard` with Day 8 `hit@1` for the same questions and say which questions moved.

## Part F: Commit
```
git add src/nlp/answer.py src/retrieval/chunker.py tools/eval_answers.py tests/test_answer.py data/eval
git commit -m "Day 10: extractive answers, spoken citations, TextRank summaries"
git log --oneline -3
```
No push.

## Acceptance criteria
1. Python 3.11.9 in `.venv`; no new packages.
2. `tests/test_answer.py`: 7 passed; Day 8 retrieval tests still 9 passed; full-suite total matches Part C.
3. Spot checks in Part C behave as listed.
4. Eval reported with `heard`, `src3`, `wrong_abstain`; targets (not gates): **heard >= 0.75**, all 5 unanswerable questions abstained. Report honestly if lower.
5. Commit made locally.

## Report back (paste verbatim, nothing more)
`sys.version` + executable | Part A | Part B diff | Part C | Part E | Part F `git log` | Deviations.
Finish with: `Done: <1 line> | Files: <list> | Next (optional): <1 line>`

## Known limitations (accepted)
- Extractive only: it reads real sentences and cannot paraphrase, combine facts across pages or reason. Yes/no questions get "The closest text says, on page N: ..." with no verdict, by design.
- Sentence picking is lexical (idf overlap, retrieval score, type cues, previous-sentence context for pronouns). Questions that share no words with the answer sentence will abstain or pick a neighbour.
- Answers depend on chunk quality (multi-column order, tables as rows, OCR noise).
- Who/when/where cues are simple patterns (capitalised words, years/months, place words); number questions prefer sentences containing digits.
- Summary quality is basic TextRank: no redundancy removal, picks 5 sentences (3 for a named section).