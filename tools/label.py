"""Day 5: build the labelled dataset for the block classifier.

  python -m tools.label data/samples/*.pdf        # extract (if needed) + label
  python -m tools.label --stats                   # counts only

Blocks are labelled automatically when a gold file (<pdf-stem>.labels.json) exists or the
block is a confident header/footer/page number. Everything else is shown to you:
Enter = accept the suggestion, or press a key to choose the label.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.tts.reader import is_heading

OUT = Path("data/labels/labels.csv")
LABELS = {"h": "heading", "p": "paragraph", "l": "list", "t": "table",
          "f": "footer", "n": "page_number", "c": "caption"}
FEATURES = ["size_ratio", "bold_ratio", "italic_ratio", "rel_x0", "rel_y0", "rel_y1", "rel_width",
            "n_chars", "n_words", "n_lines", "digit_ratio", "upper_ratio", "ends_with_period",
            "starts_with_bullet", "looks_like_page_number", "repeated_margin", "is_ocr"]
COLS = ["doc", "block_id", "label", "how", "text"] + FEATURES
CAPTION_RE = re.compile(r"^(figure|fig\.|table)\s*\d+", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def suggest(b: dict) -> tuple[str, bool]:
    """Heuristic label and whether it is confident enough to accept without asking."""
    if b["looks_like_page_number"]:
        return "page_number", True
    if b["repeated_margin"]:
        return "footer", True
    if CAPTION_RE.match(b["text"]):
        return "caption", False
    if b["starts_with_bullet"]:
        return "list", False
    return ("heading" if is_heading(b) else "paragraph"), False


def gold_label(b: dict, gold: dict) -> str | None:
    n = norm(b["text"])
    if n in gold:
        return gold[n]
    if len(n) >= 12:  # tolerate small extraction differences (bullets, merged lines)
        for k, v in gold.items():
            if len(k) >= 12 and (n in k or k in n):
                return v
    return None


def to_row(doc: str, b: dict, label: str, how: str) -> dict:
    row = {"doc": doc, "block_id": b["id"], "label": label, "how": how, "text": b["text"][:80]}
    for f in FEATURES:
        row[f] = int(b.get("source") == "ocr") if f == "is_ocr" else b[f]
    return row


def load_rows() -> list[dict]:
    return list(csv.DictReader(OUT.open(encoding="utf-8"))) if OUT.exists() else []


def save_rows(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def read_key() -> str:
    try:
        import msvcrt
        return msvcrt.getwch()
    except ImportError:
        return (input("> ").strip()[:1] or "\r")


def stats(rows: list[dict]) -> None:
    print(f"{len(rows)} labelled blocks in {len({r['doc'] for r in rows})} documents")
    print("by label:", dict(Counter(r["label"] for r in rows)))
    print("by how:  ", dict(Counter(r["how"] for r in rows)))


def get_doc(path: Path) -> dict:
    js = Path("data/processed") / (path.stem + ".json")
    if path.suffix.lower() == ".pdf" and not js.exists():
        from src.ingestion.extractor import extract_document, save_json
        save_json(extract_document(path))
    return json.loads(js.read_text(encoding="utf-8"))


def label_file(path: Path, rows: list[dict]) -> None:
    doc = get_doc(path)
    gold_path = path.with_suffix("").with_suffix(".labels.json")
    gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else {}
    done = {(r["doc"], r["block_id"]) for r in rows}
    for page in doc["pages"]:
        for b in page["blocks"]:
            if (doc["source"], b["id"]) in done:
                continue
            g = gold_label(b, gold)
            sug, confident = suggest(b)
            if g or confident:
                rows.append(to_row(doc["source"], b, g or sug, "gold" if g else "auto"))
                continue
            print(f"\n[{doc['source']} p{b['page']}] size={b['size_ratio']}x bold={b['bold_ratio']} "
                  f"y={b['rel_y0']} words={b['n_words']}\n  {b['text'][:200]}\n  suggestion: {sug}  "
                  f"[h]eading [p]aragraph [l]ist [t]able [f]ooter page-[n]umber [c]aption "
                  f"Enter=accept u=undo s=skip q=quit")
            while True:
                k = read_key().lower()
                if k in ("\r", "\n", " "):
                    rows.append(to_row(doc["source"], b, sug, "manual")); break
                if k in LABELS:
                    rows.append(to_row(doc["source"], b, LABELS[k], "manual")); break
                if k == "u" and rows:
                    print(f"  undone: {rows.pop()['text'][:40]}"); save_rows(rows)
                if k == "s":
                    break
                if k == "q":
                    save_rows(rows); raise SystemExit(0)
            save_rows(rows)
    save_rows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    rows = load_rows()
    for f in args.files:
        label_file(Path(f), rows)
    stats(rows)


if __name__ == "__main__":
    main()
