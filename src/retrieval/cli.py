"""python -m src.retrieval.cli <pdf|json> "question" [-k 3] [--classify]"""
import argparse
import json
from pathlib import Path

from src.retrieval.index import Index

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("question")
p.add_argument("-k", type=int, default=3)
p.add_argument("--classify", action="store_true", help="use the block classifier if its model exists")
a = p.parse_args()

if a.source.lower().endswith(".json"):
    doc = json.load(open(a.source, encoding="utf-8"))
else:
    from src.ingestion.extractor import extract_document
    doc = extract_document(a.source)
if a.classify:
    from src.nlp.block_classifier import MODEL_PATH, classify_document
    if Path(MODEL_PATH).exists():
        classify_document(doc)
    else:
        print("classifier model not found; using heuristics")
idx = Index.from_doc(doc)
print(f"{len(idx.chunks)} chunks")
for r in idx.query(a.question, a.k):
    print(f"[p.{r['page']}] {r['heading'] or '-'} (score {r['score']}): {r['text'][:200]}")

