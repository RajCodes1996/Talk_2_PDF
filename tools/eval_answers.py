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

