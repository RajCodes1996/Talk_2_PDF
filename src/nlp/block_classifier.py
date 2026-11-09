"""Block classifier: predicts heading/paragraph/list/table/footer/page_number/caption."""
import csv
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


def load_labels(csv_path):
    """Returns X, y, docs, how, rows (rows kept for baseline comparison)."""
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] in LABELS]
    X = np.array([[_f(r[k]) for k in FEATURES] for r in rows])
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


def train(csv_path, out_path=MODEL_PATH, log=print):
    """Grouped-by-document CV picks the model; final model is fit on all rows."""
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X, y, docs, how, rows = load_labels(csv_path)
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
    log(f"docs={n_docs} rows={len(y)} macro-F1 by model: {report} -> chose {best}")
    log(classification_report(y, best_pred, zero_division=0))
    model = make_models()[best].fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "labels": LABELS, "name": best}, out_path)
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
    m = load_model(model_path)["model"]
    X = np.array([block_features(b) for b in blocks])
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

