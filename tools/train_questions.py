"""Train the question-type classifier and evaluate on heldout.

Usage: python -m tools.train_questions

Saves model to data/models/question_clf.joblib and log to data/models/question_log.txt.
"""

import numpy as np
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.nlp.question_classifier import LABELS, MODEL_PATH, load_csv, make_pipeline, train

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))

# Load training and heldout data
Xtr, ytr = load_csv("data/questions/train.csv")
Xh, yh = load_csv("data/questions/heldout.csv")
if Path("data/questions/heldout_user.csv").exists():
    xu, yu = load_csv("data/questions/heldout_user.csv")
    Xh, yh = Xh + xu, yh + yu
    log(f"including {len(xu)} user-written heldout questions")
log(f"train={len(Xtr)} heldout={len(Xh)} counts={dict(Counter(ytr))}")

# Cross‑validation on training set (optimistic)
cv = cross_val_predict(make_pipeline(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42))
log(f"5‑fold CV on train (optimistic, templated data): macro‑F1={f1_score(ytr, cv, average='macro'):.3f}")

# Train final model
model = train("data/questions/train.csv", MODEL_PATH)
pred = list(model.predict(Xh))
log(f"HELDOUT macro‑F1={f1_score(yh, pred, average='macro'):.3f} accuracy={np.mean(np.array(pred) == np.array(yh)):.3f}")
log(classification_report(yh, pred, labels=LABELS, zero_division=0))
cm = confusion_matrix(yh, pred, labels=LABELS)
log("confusion (rows=true, cols=pred): " + " ".join(l[:5] for l in LABELS))
for l, r in zip(LABELS, cm):
    log(f"{l[:9]:>9} " + " ".join(f"{v:5d}" for v in r))

# Margin analysis
S = model.decision_function(Xh)
srt = np.sort(S, axis=1)
margin = srt[:, -1] - srt[:, -2]
log("margin threshold -> coverage, accuracy on kept:")
for t in (0.0, 0.1, 0.2, 0.3, 0.5):
    keep = margin >= t
    acc = np.mean(np.array(pred)[keep] == np.array(yh)[keep]) if keep.any() else 0
    log(f"  {t}: coverage={keep.mean():.2f} acc={acc:.3f}")
log("misclassified (true -> pred | margin | text):")
for x, t, p, m in zip(Xh, yh, pred, margin):
    if t != p:
        log(f"  {t} -> {p} | {m:.2f} | {x}")

Path("data/models/question_log.txt").write_text("\n".join(lines), encoding="utf-8")

