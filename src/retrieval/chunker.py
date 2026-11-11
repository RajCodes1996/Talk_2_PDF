"""Split an extractor-JSON document into retrievable chunks with page metadata."""
import re

SKIP_LABELS = {"footer", "page_number"}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _skip(b):
    if "label" in b:
        return b["label"] in SKIP_LABELS
    return bool(b.get("looks_like_page_number") or b.get("repeated_margin"))


def _heading(b):
    if "label" in b:
        return b["label"] == "heading"
    return (b.get("size_ratio", 1) >= 1.2 and b.get("n_words", 99) <= 15
            and not b.get("ends_with_period") and b.get("n_lines", 1) <= 2)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT.split(text) if s]


def build_chunks(doc, max_words=120, min_words=15):
    """Chunks follow reading order; a heading starts a new chunk (and is carried across pages).
    Each chunk: id, heading, heading_id, text (body only), page_start, page_end, block_ids, n_words."""
    chunks, cur = [], None
    head, head_id = "", None
    body_seen = True  # False right after a heading: consecutive headings are merged into one

    def flush():
        nonlocal cur
        if cur and cur["sents"]:
            chunks.append({
                "id": f"c{len(chunks)}", "heading": cur["heading"], "heading_id": cur["heading_id"],
                "text": " ".join(cur["sents"]), "page_start": cur["pages"][0],
                "page_end": cur["pages"][-1], "block_ids": sorted(set(cur["bids"]), key=cur["bids"].index),
                "n_words": cur["n"]})
        cur = None

    def new():
        return {"heading": head, "heading_id": head_id, "sents": [], "pages": [], "bids": [], "n": 0}

    for page in doc["pages"]:
        for b in page["blocks"]:
            text = (b.get("text") or "").strip()
            if not text or _skip(b):
                continue
            if _heading(b):
                flush()
                t = re.sub(r"\s+", " ", text)
                if body_seen:
                    head, head_id = t, b.get("id")
                else:  # e.g. column headings listed before their bodies (sam.pdf)
                    head = f"{head}; {t}"
                body_seen = False
                continue
            for s in split_sentences(text):
                w = len(s.split())
                body_seen = True
                if cur is None:
                    cur = new()
                if cur["n"] >= min_words and cur["n"] + w > max_words:
                    flush()
                    cur = new()
                cur["sents"].append(s)
                cur["pages"].append(b["page"])
                cur["bids"].append(b.get("id"))
                cur["n"] += w
    flush()
    # merge tiny leftovers into the previous chunk of the same section
    out = []
    for c in chunks:
        if out and c["n_words"] < min_words and out[-1]["heading_id"] == c["heading_id"]:
            p = out[-1]
            p["text"] += " " + c["text"]
            p["page_end"] = c["page_end"]
            p["block_ids"] += [x for x in c["block_ids"] if x not in p["block_ids"]]
            p["n_words"] += c["n_words"]
        else:
            out.append(c)
    for i, c in enumerate(out):
        c["id"] = f"c{i}"
    return out