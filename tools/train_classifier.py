"""python -m tools.train_classifier [--csv data/labels/labels.csv]
Trains, prints per-class report, compares with Reader heuristics, logs to data/models/train_log.txt"""
import argparse
from collections import Counter

from src.nlp.block_classifier import MODEL_PATH, train

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/labels/labels.csv")
a = p.parse_args()

lines = []


def log(s):
    print(s)
    lines.append(str(s))


res = train(a.csv, MODEL_PATH, log)
rows, y, pred = res["rows"], res["y"], res["pred"]
log(f"label counts: {dict(Counter(y))}")

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

