# task_day7.md: Buffer day (diagnose, error report, optional context features, commit)

Goal: close HANDOVER open items 1 and 2, retrain on the real labels, pick the best variant by measured F1, commit. Nothing here adds product features beyond the optional context switch.

## Rules (override AGENTS.md where they conflict)
- Work in `C:\Users\netware\Desktop\vision`, run from project root only.
- FIRST: `python -c "import sys; print(sys.version); print(sys.executable)"` must show **3.11.9** and a path inside `vision\.venv`. Otherwise STOP and report.
- Forbidden: uv/poetry/conda, recreating `.venv`, creating `pyproject.toml`/`uv.lock`, `import fitz`. Use `python -m pytest`, `python -m pip`.
- Edit ONLY the files listed in Part B. Do NOT edit `tests/test_ocr.py`, `tools/label.py` or any other file. Do NOT revert anything found in Part A; report it.
- AGENTS.md "narrowest test" rule is waived for the two full-suite runs (Part A3, Part E) because the task requires them. Everywhere else keep output short: pipe to `Select-Object -Last N` as written. Never paste more than the requested lines.
- Interactive labelling (`tools.label`) is done by the user, never by you.
- Report REAL output. "Deviations" must list every file you touched or created that is not named here.
- If a step fails twice: stop, report the last 20 lines of the error plus 2 options.

## Part A: Diagnose (read-only, no code changes)
A1. Python check above.
A2. `python -m pytest --collect-only -q | Select-Object -Last 35`
    Expected after Day 6: **24 tests** (2 smoke, 3 extractor, 6 OCR, 8 reader, 3 labeling, 2 classifier). Report per-file counts.
A3. `python -m pytest -q -rs --durations=5 | Select-Object -Last 20`
    Expected: all pass, 10 to 40 s total (earlier runs were 10 to 40 s; one report said 330 s with 16 passed). Report the summary line, any SKIPPED reasons, and the 5 slowest tests.
A4. Side effects:
```
git status --short
git log --oneline -5
Test-Path pyproject.toml, uv.lock
rg -n "CAPTION_RE" tools/label.py
```
    Expected: `Test-Path` returns False, False. Report what `CAPTION_RE` is now (original was `^(figure|fig\.|table)\s*\d+`).
If A2 or A3 shows missing/skipped OCR tests or a very slow run, do NOT fix: report the cause (skip reason, Tesseract path, slowest test) plus 2 options, then continue with Part B.

## Part B: Replace two files, extend the tests
Prerequisite: `src/nlp/block_classifier.py` from Day 6 exists. If not, STOP.
Changes: `load_labels`/`train` gain `manual_only` and `context` options (context = previous/next block features, off by default); `predict` reads the `context` flag stored in the model file; `tools/train_classifier.py` gains `--manual-only`, `--context`, `--out`, a confusion matrix and `data/models/errors.csv` (misclassified rows).

### `src/nlp/block_classifier.py` (full replacement)
```python
"""Block classifier: predicts heading/paragraph/list/table/footer/page_number/caption."""
import csv
import re
from pathlib import Path

import joblib
import numpy as np

LABELS = ["heading", "paragraph", "list", "table", "footer", "page_number", "caption"]
FEATURES = ["size_ratio", "bold_ratio", "italic_ratio", "rel_x0", "rel_y0", "rel_y1",
            "rel_width", "n_chars", "n_words", "n_lines", "digit_ratio", "upper_ratio",
            "ends_with_period", "starts_with_bullet", "looks_like_page_number",
            "repeated_margin", "is_ocr"]
MODEL_PATH = Path("data/models/block_classifier.joblib")
SEED = 42


def _f(v):
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in ("true", "false"):
            return 1.0 if v.lower() == "true" else 0.0
        return float(v) if v else 0.0
    return float(v or 0)


def block_features(block):
    row = dict(block)
    row.setdefault("is_ocr", block.get("source") == "ocr")
    return [_f(row.get(k, 0)) for k in FEATURES]


CTX_KEYS = ["size_ratio", "rel_y0", "n_words", "starts_with_bullet", "bold_ratio"]
_CTX_IDX = [FEATURES.index(k) for k in CTX_KEYS]


def add_context(X, groups):
    """Append prev/next-block features (+ is_first/is_last flags). Rows must be in reading
    order with each group (doc, page) contiguous. Neighbours are taken within the group."""
    n, k = len(X), len(_CTX_IDX)
    prev, nxt = np.zeros((n, k + 1)), np.zeros((n, k + 1))
    for i in range(n):
        if i > 0 and groups[i - 1] == groups[i]:
            prev[i, :k] = X[i - 1, _CTX_IDX]
        else:
            prev[i, k] = 1
        if i < n - 1 and groups[i + 1] == groups[i]:
            nxt[i, :k] = X[i + 1, _CTX_IDX]
        else:
            nxt[i, k] = 1
    return np.hstack([X, prev, nxt])


def _order(r, i):
    m = re.match(r"p(\d+)_b(\d+)", r["block_id"])
    return (r["doc"], int(m.group(1)), int(m.group(2))) if m else (r["doc"], 0, i)


def load_labels(csv_path, manual_only=False, context=False):
    """Returns X, y, docs, how, rows (sorted in reading order; rows kept for error reports)."""
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] in LABELS]
    if manual_only:
        rows = [r for r in rows if r["how"] in ("gold", "manual")]
    rows = [r for _, r in sorted(((_order(r, i), r) for i, r in enumerate(rows)),
                                 key=lambda t: t[0])]
    X = np.array([[_f(r[k]) for k in FEATURES] for r in rows])
    if context:
        X = add_context(X, [_order(r, 0)[:2] for r in rows])
    return (X, np.array([r["label"] for r in rows]), np.array([r["doc"] for r in rows]),
            np.array([r["how"] for r in rows]), rows)


def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=1),
    }


def train(csv_path, out_path=MODEL_PATH, log=print, manual_only=False, context=False):
    """Grouped-by-document CV picks the model; final model is fit on all rows."""
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X, y, docs, how, rows = load_labels(csv_path, manual_only, context)
    n_docs = len(set(docs))
    if n_docs < 2:
        raise ValueError("Need at least 2 documents for a by-document split")
    cv = GroupKFold(n_splits=min(5, n_docs))
    best, best_f1, best_pred, report = None, -1, None, {}
    for name, m in make_models().items():
        pred = cross_val_predict(m, X, y, groups=docs, cv=cv)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        report[name] = round(f1, 3)
        if f1 > best_f1:
            best, best_f1, best_pred = name, f1, pred
    log(f"docs={n_docs} rows={len(y)} manual_only={manual_only} context={context} macro-F1 by model: {report} -> chose {best}")
    log(classification_report(y, best_pred, zero_division=0))
    model = make_models()[best].fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "labels": LABELS, "name": best,
                 "context": context}, out_path)
    return {"best": best, "macro_f1": best_f1, "pred": best_pred, "y": y, "rows": rows}


_cache = {}


def load_model(path=MODEL_PATH):
    p = str(path)
    if p not in _cache:
        _cache[p] = joblib.load(p)
    return _cache[p]


def predict(blocks, model_path=MODEL_PATH):
    """Adds 'label' and 'label_conf' to each block dict (in place) and returns the list."""
    if not blocks:
        return blocks
    bundle = load_model(model_path)
    m = bundle["model"]
    X = np.array([block_features(b) for b in blocks])
    if bundle.get("context"):  # blocks = one page, in reading order
        X = add_context(X, [0] * len(blocks))
    labels = m.predict(X)
    conf = m.predict_proba(X).max(axis=1) if hasattr(m, "predict_proba") else [1.0] * len(blocks)
    for b, l, c in zip(blocks, labels, conf):
        b["label"], b["label_conf"] = str(l), round(float(c), 3)
    return blocks


def classify_document(doc, model_path=MODEL_PATH):
    """Annotate every block of an extractor-JSON dict; Reader then uses block['label']."""
    for page in doc["pages"]:
        predict(page["blocks"], model_path)
    return doc
```
### `tools/train_classifier.py` (full replacement)
```python
"""python -m tools.train_classifier [--csv PATH] [--manual-only] [--context] [--out PATH]
Trains (grouped-by-doc CV), prints per-class report + confusion matrix, compares with Reader
heuristics, writes data/models/train_log.txt and data/models/errors.csv (misclassified rows)."""
import argparse
import csv
from collections import Counter

from src.nlp.block_classifier import LABELS, MODEL_PATH, train

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/labels/labels.csv")
p.add_argument("--manual-only", action="store_true", help="use only gold+manual rows")
p.add_argument("--context", action="store_true", help="add prev/next block features")
p.add_argument("--out", default=str(MODEL_PATH))
a = p.parse_args()

lines = []


def log(s):
    print(s)
    lines.append(str(s))


res = train(a.csv, a.out, log, manual_only=a.manual_only, context=a.context)
rows, y, pred = res["rows"], res["y"], res["pred"]
log(f"label counts: {dict(Counter(map(str, y)))}")

try:  # confusion matrix (rows=true, cols=pred)
    from sklearn.metrics import confusion_matrix
    labs = [l for l in LABELS if l in set(y) | set(pred)]
    cm = confusion_matrix(y, pred, labels=labs)
    log("confusion (rows=true, cols=pred): " + " ".join(l[:5] for l in labs))
    for l, r in zip(labs, cm):
        log(f"{l[:9]:>9} " + " ".join(f"{v:5d}" for v in r))
except Exception as e:
    log(f"confusion matrix skipped: {e}")

with open("data/models/errors.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["doc", "block_id", "true", "pred", "how", "text"])
    for r, t, q in zip(rows, y, pred):
        if t != q:
            w.writerow([r["doc"], r["block_id"], t, q, r["how"], r["text"][:80]])
log(f"misclassified: {int((y != pred).sum())} of {len(y)} -> data/models/errors.csv")

try:  # Baseline: Reader heuristics vs gold on the same rows
    from sklearn.metrics import f1_score
    from src.tts.reader import is_heading, is_skipped

    def blk(r):
        d = {}
        for k, v in r.items():
            if k in ("text", "doc", "label", "how", "block_id"):
                d[k] = v
            elif v in ("True", "False"):
                d[k] = v == "True"
            else:
                d[k] = float(v or 0)
        d["id"] = r["block_id"]
        d["source"] = "ocr" if d.get("is_ocr") else "text"
        return d

    bl = [blk(r) for r in rows]
    h_t = [l == "heading" for l in y]
    s_t = [l in ("footer", "page_number") for l in y]
    log(f"heading F1  heuristic={f1_score(h_t, [bool(is_heading(b)) for b in bl]):.3f}"
        f"  ml={f1_score(h_t, [l == 'heading' for l in pred]):.3f}")
    log(f"skip F1     heuristic={f1_score(s_t, [bool(is_skipped(b)) for b in bl]):.3f}"
        f"  ml={f1_score(s_t, [l in ('footer', 'page_number') for l in pred]):.3f}")
except Exception as e:
    log(f"baseline comparison skipped: {type(e).__name__}: {e}")

open("data/models/train_log.txt", "w", encoding="utf-8").write("\n".join(lines))
```
### `tests/test_classifier.py` (full replacement)
```python
import csv

import numpy as np
import pytest

from src.nlp.block_classifier import FEATURES, LABELS, classify_document, predict, train


def _synth(path, n_docs=4, per=42, seed=0):
    rng = np.random.default_rng(seed)
    protos = {  # size_ratio, bold, rel_y0, n_words, bullet, pagenum, repeated
        "heading": (1.6, .9, .2, 4, 0, 0, 0), "paragraph": (1.0, 0, .5, 60, 0, 0, 0),
        "list": (1.0, 0, .5, 12, 1, 0, 0), "table": (.95, 0, .5, 5, 0, 0, 0),
        "footer": (.8, 0, .95, 6, 0, 0, 1), "page_number": (.8, 0, .97, 1, 0, 1, 1),
        "caption": (.9, 0, .6, 10, 0, 0, 0)}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["doc", "block_id", "label", "how", "text"] + FEATURES)
        w.writeheader()
        for d in range(n_docs):
            for i in range(per):
                lab = LABELS[i % 7]
                s, b, y0, nw, bu, pn, rp = protos[lab]
                r = dict.fromkeys(FEATURES, 0)
                r.update(size_ratio=s + rng.normal(0, .03), bold_ratio=b,
                         rel_y0=y0 + rng.normal(0, .02), rel_y1=y0 + .03,
                         n_words=nw + int(rng.integers(0, 3)), n_chars=(nw + 2) * 6,
                         starts_with_bullet=bu, looks_like_page_number=pn, repeated_margin=rp,
                         rel_x0=.1, rel_width=.8, n_lines=1, is_ocr=0)
                w.writerow({"doc": f"d{d}", "block_id": f"p1_b{i}", "label": lab,
                            "how": "gold", "text": "x", **r})


def test_train_predict_and_doc(tmp_path):
    c, m = tmp_path / "l.csv", tmp_path / "m.joblib"
    _synth(c)
    res = train(c, m, log=lambda s: None)
    assert res["macro_f1"] > 0.8
    block = {"size_ratio": 1.6, "bold_ratio": .9, "rel_y0": .2, "n_words": 4, "n_chars": 36,
             "rel_y1": .23, "rel_x0": .1, "rel_width": .8, "n_lines": 1, "source": "text"}
    out = predict([dict(block)], m)
    assert out[0]["label"] == "heading" and 0 < out[0]["label_conf"] <= 1
    doc = classify_document({"pages": [{"blocks": [dict(block), dict(block)]}]}, m)
    assert all("label" in b for b in doc["pages"][0]["blocks"])


def test_needs_two_docs(tmp_path):
    c = tmp_path / "l.csv"
    _synth(c, n_docs=1)
    with pytest.raises(ValueError):
        train(c, tmp_path / "m.joblib", log=lambda s: None)


def test_context_and_manual_only(tmp_path):
    from src.nlp.block_classifier import add_context, load_labels
    c, m = tmp_path / "l.csv", tmp_path / "m.joblib"
    _synth(c)
    X, y, docs, how, rows = load_labels(c, context=True)
    assert X.shape[1] == len(FEATURES) + 2 * 6 and len(X) == len(y)
    assert add_context(np.ones((3, len(FEATURES))), [0, 0, 1]).shape == (3, len(FEATURES) + 12)
    assert len(load_labels(c, manual_only=True)[1]) == len(y)  # gold rows are kept
    res = train(c, m, log=lambda s: None, context=True)
    assert res["macro_f1"] > 0.8
    block = {"size_ratio": 1.6, "bold_ratio": .9, "rel_y0": .2, "n_words": 4, "n_chars": 36,
             "rel_y1": .23, "rel_x0": .1, "rel_width": .8, "n_lines": 1, "source": "text"}
    assert predict([dict(block), dict(block)], m)[0]["label"] in LABELS
```
Verify: `python -m pytest tests/test_classifier.py -q` → expected **3 passed** (about 4 s).

## Part C: Train and compare (uses the user's real `data/labels/labels.csv`)
Check first: `python -m tools.label --stats | Select-Object -Last 15`. Report the manual-label counts. If manual labels < 350, STOP and tell the user to label more.
Run the three variants (each writes its own model file, log goes to chat only as the lines below):
```
python -m tools.train_classifier --out data\models\v_all.joblib
python -m tools.train_classifier --manual-only --out data\models\v_manual.joblib
python -m tools.train_classifier --manual-only --context --out data\models\v_manual_ctx.joblib
```
After each run report ONLY: the `macro-F1 by model` line, the per-class table (precision/recall/F1/support), the confusion matrix, the `misclassified:` line, and the two baseline lines (heading F1, skip F1). Put the three results in one table: variant | chosen model | macro-F1 | heading F1 | list F1 | table F1 | caption F1.
Decision rule: choose the highest macro-F1. Prefer the simpler variant unless the more complex one gains at least 0.02 macro-F1. Note: `v_all` includes noisy `auto` rows and a different row set, so it is not strictly comparable; do not pick it just for a higher number if its manual-only per-class scores are lower.
`--context` caveat: neighbours are taken from adjacent labelled rows, so it only helps if documents are fully labelled. State in the report whether labelled blocks per page look complete (compare a doc's row count with its block count in `data\processed\*.json`, one doc only).
Retrain the winner to the default path (rerun the winning command with no `--out`) → `data\models\block_classifier.joblib`. Then: `Get-Content data\models\errors.csv -TotalCount 15` and report it.
Delete the three `v_*.joblib` files afterwards.

## Part D: End-to-end
`python -m src.tts.cli data\samples\sam.pdf --dry-run --classify | Select-Object -First 15`
Expected: runs without error, headings announced, footers/page numbers skipped. Paste the 15 lines.

## Part E: Commit (only if everything above passed)
1. `python -m pytest -q | Select-Object -Last 5` → expected **25 passed**. If not 25 or any failure: do NOT commit; report.
2. Confirm `.gitignore` contains `.venv` (add the single line if missing and say so).
3. `git add src tests tools data/labels data/models/block_classifier.joblib data/models/train_log.txt requirements.txt pytest.ini .gitignore`
4. `git commit -m "Day 6-7: block classifier, error report, optional context features"` (no push). Report `git log --oneline -3` and `git status --short`.

## Acceptance criteria
1. Python 3.11.9 in `.venv`; no `pyproject.toml`/`uv.lock`.
2. Part A numbers reported with real output; the 6-test OCR discrepancy explained or flagged with 2 options.
3. Classifier tests: 3 passed; full suite: 25 passed.
4. Three-variant table delivered; winner justified by the decision rule; model saved at the default path.
5. Per-class F1 reported honestly. Target (not a gate): >= 0.90 on heading, paragraph, footer, page_number.
6. `--classify --dry-run` works; commit made locally.

## Report back (paste verbatim, nothing more)
`sys.version` + executable | A2 counts by file | A3 summary + skips + 5 slowest | A4 outputs | Part C table + winner's confusion matrix + first 15 errors | Part D lines | Part E `git log`/`git status` | Deviations (files touched beyond the list).

## Known limitations
- Per-block classifier; context mode only sees one neighbour each side.
- If list/table/caption F1 stays low (< 0.80), the fix is more labelled examples of those classes, not more features: use `data\models\errors.csv` to see which documents and layouts fail, then label similar pages and retrain.
- Scanned pages have `bold_ratio = 0` and estimated sizes, so heading recall on OCR documents is expected to be weaker.