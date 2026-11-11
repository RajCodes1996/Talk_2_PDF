"""Question-type classifier (TF‑IDF word+char n‑grams + LinearSVC) and page-number parsing.

This module provides:
- `train(train_csv, out_path=MODEL_PATH)` – train and persist a scikit‑learn pipeline.
- `load_model(path=MODEL_PATH)` – cached model loader.
- `predict(question, model=None, min_margin=None)` – returns (label, margin).
- `extract_page(q)` – extracts a page number from a question.
- `plan(question, model=None, min_margin=None)` – combines the above into a routing dict.
"""

import csv
import re
from pathlib import Path

import joblib
import numpy as np

LABELS = ["definition", "wh_fact", "number", "yes_no", "summary", "page_lookup", "other"]
MODEL_PATH = Path("data/models/question_clf.joblib")
MIN_MARGIN = 0.1  # below this top1‑top2 margin the type falls back to "other"
# Day 10 handlers keyed by type (names only; handlers come on Day 10)
ROUTES = {
    "definition": "define",
    "wh_fact": "fact",
    "number": "number",
    "yes_no": "yes_no",
    "summary": "summarize",
    "page_lookup": "page",
    "other": "passages",
}

_NUMW = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split()
    )
}
_ORD = {w: i + 1 for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth".split())}


def normalize(q):
    return re.sub(r"\\d+", " 0 ", q.lower().strip())


def make_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline as mk
    from sklearn.svm import LinearSVC

    feats = FeatureUnion([
        (
            "w",
            TfidfVectorizer(
                preprocessor=normalize,
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "c",
            TfidfVectorizer(
                preprocessor=normalize,
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
            ),
        ),
    ])
    return mk(feats, LinearSVC(C=0.5, class_weight="balanced", random_state=42))


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] in LABELS and r["text"].strip()]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def train(train_csv, out_path=MODEL_PATH):
    X, y = load_csv(train_csv)
    m = make_pipeline().fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, out_path)
    return m

_cache = {}


def load_model(path=MODEL_PATH):
    if str(path) not in _cache:
        _cache[str(path)] = joblib.load(path)
    return _cache[str(path)]


def predict(question, model=None, min_margin=None):
    """-> (label, margin). Low margin falls back to 'other'."""
    m = model or load_model()
    mm = MIN_MARGIN if min_margin is None else min_margin
    s = m.decision_function([question])[0]
    order = np.argsort(-s)
    margin = float(s[order[0]] - s[order[1]])
    label = str(m.classes_[order[0]])
    return (label if margin >= mm else "other"), round(margin, 3)


def extract_page(q):
    """Page number named in a question: int, -1 for 'last page', None if none found."""
    t = q.lower()
    if re.search(r"\\blast page\\b|\\bfinal page\\b", t):
        return -1
    m = re.search(r"\\bpage\\s*(?:no\\.?|number|#)?\\s*(\\d+)\\b|\\bp\\.?\\s*(\\d+)\\b", t)
    if m:
        return int(m.group(1) or m.group(2))
    m = re.search(r"\\bpage\\s+(" + "|".join(_NUMW) + r")\\b", t)
    if m:
        return _NUMW[m.group(1)]
    m = re.search(r"\\b(" + "|".join(_ORD) + r")\\s+page\\b", t)
    return _ORD[m.group(1)] if m else None


def plan(question, model=None, min_margin=None):
    label, margin = predict(question, model, min_margin)
    page = extract_page(question)
    if label == "page_lookup" and page is None:
        label = "other"
    return {"type": label, "margin": margin, "page": page, "handler": ROUTES[label]}

if __name__ == "__main__":
    import sys
    print(plan(" ".join(sys.argv[1:])))

