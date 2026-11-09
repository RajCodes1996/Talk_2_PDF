"""Hybrid retrieval: BM25 + TF-IDF cosine over chunks, fully offline."""
import re

import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.retrieval.chunker import build_chunks

STOP = set("""a an the and or of to in on at for from by with as is are was were be been being it its this that these
those which who whom what when where why how do does did can could should would will shall may might i you he she
we they them his her their our your not no than then so if but about into over under also any all some""".split())
_TOK = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def stem(w):
    if w.isdigit() or len(w) <= 3:
        return w
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


def tokenize(text):
    toks = (re.sub(r"'s$", "", t) for t in _TOK.findall(text.lower()))
    return [stem(t) for t in toks if t not in STOP]


def _norm(x):
    m = x.max() if len(x) else 0
    return x / m if m > 0 else np.zeros_like(x)


class Index:
    def __init__(self, chunks, alpha=0.5):
        """alpha = weight of BM25 (1-alpha = TF-IDF). Heading text is indexed with its chunk."""
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks, self.alpha = chunks, alpha
        self.tokens = [tokenize(f"{c['heading']} {c['text']}") for c in chunks]
        if not any(self.tokens):
            raise ValueError("Chunks contain no indexable words")
        self.bm25 = BM25Okapi([t or ["_"] for t in self.tokens])
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     sublinear_tf=True, ngram_range=(1, 2))
        self.matrix = self.tfidf.fit_transform([f"{c['heading']} {c['text']}" for c in chunks])

    @classmethod
    def from_doc(cls, doc, alpha=0.5, **chunk_kw):
        return cls(build_chunks(doc, **chunk_kw), alpha)

    def query(self, q, k=3, alpha=None):
        a = self.alpha if alpha is None else alpha
        toks = tokenize(q)
        if not toks:
            return []
        bm = _norm(np.asarray(self.bm25.get_scores(toks), dtype=float).clip(min=0))
        tf = _norm(linear_kernel(self.tfidf.transform([q]), self.matrix).ravel())
        score = a * bm + (1 - a) * tf
        order = [i for i in np.argsort(-score, kind="stable") if score[i] > 0][:k]
        return [{"chunk_id": self.chunks[i]["id"], "score": round(float(score[i]), 4),
                 "bm25": round(float(bm[i]), 4), "tfidf": round(float(tf[i]), 4),
                 "page": self.chunks[i]["page_start"], "page_end": self.chunks[i]["page_end"],
                 "heading": self.chunks[i]["heading"], "text": self.chunks[i]["text"]} for i in order]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

