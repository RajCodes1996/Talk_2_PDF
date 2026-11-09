"""python -m tools.eval_retrieval [--csv data/eval/retrieval_eval.csv] [--classify] [--alpha 0.5]
CSV columns: doc,question,expected_phrase,expected_page (phrase or page may be empty, not both).
A hit = a returned chunk contains expected_phrase (case-insensitive), else its page range holds expected_page.
Reports hit@1, hit@3, MRR (and per-alpha sweep with --sweep). Writes data/eval/results.txt."""
import argparse
import csv
import json
from pathlib import Path

from src.retrieval.index import Index

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/eval/retrieval_eval.csv")
p.add_argument("--processed", default="data/processed")
p.add_argument("--samples", default="data/samples")
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--sweep", action="store_true")
p.add_argument("--classify", action="store_true")
p.add_argument("--out", default="data/eval/results.txt")
a = p.parse_args()


def load_doc(name):
    j = Path(a.processed) / f"{name}.json"
    if j.exists():
        return json.load(open(j, encoding="utf-8"))
    from src.ingestion.extractor import extract_document
    return extract_document(str(Path(a.samples) / f"{name}.pdf"))


rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if r["question"].strip()]
docs, idx = {}, {}
for r in rows:
    if r["doc"] not in docs:
        d = load_doc(r["doc"])
        if a.classify:
            from src.nlp.block_classifier import classify_document
            d = classify_document(d)
        docs[r["doc"]] = d
        idx[r["doc"]] = Index.from_doc(d)


def hit(res, r):
    ph, pg = r["expected_phrase"].strip().lower(), r["expected_page"].strip()
    if ph:
        return ph in res["text"].lower() or ph in res["heading"].lower()
    return bool(pg) and res["page"] <= int(pg) <= res["page_end"]


def run(alpha):
    h1 = h3 = mrr = 0.0
    misses = []
    for r in rows:
        res = idx[r["doc"]].query(r["question"], 3, alpha)
        rank = next((i + 1 for i, x in enumerate(res) if hit(x, r)), 0)
        h1 += rank == 1
        h3 += rank > 0
        mrr += 1 / rank if rank else 0
        if rank != 1:
            misses.append((r["doc"], r["question"], rank, res[0]["page"] if res else None))
    n = len(rows)
    return h1 / n, h3 / n, mrr / n, misses


lines = [f"questions={len(rows)} docs={len(docs)} chunks={ {d: len(i.chunks) for d, i in idx.items()} }"]
for al in ([0.0, 0.3, 0.5, 0.7, 1.0] if a.sweep else [a.alpha]):
    h1, h3, m, misses = run(al)
    lines.append(f"alpha={al}: hit@1={h1:.3f} hit@3={h3:.3f} MRR={m:.3f}")
lines.append(f"not at rank 1, alpha={al} (doc | question | rank | top1 page):")
lines += [f"  {d} | {q[:60]} | {rk} | {pg}" for d, q, rk, pg in misses]
print("\n".join(lines))
Path(a.out).write_text("\n".join(lines), encoding="utf-8")

