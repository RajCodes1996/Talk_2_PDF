"""Structure-aware text extraction from PDFs (fully offline).

Day 2: digital PDFs -> blocks with layout features.
Day 3: scanned pages are detected and OCR'd (Tesseract) into the SAME block schema.

Output schema (JSON):
{
  "source": "file.pdf", "num_pages": int, "body_font_size": float,
  "ocr_pages": [page numbers that went through OCR],
  "pages": [
    {"page": int, "width": float, "height": float, "needs_ocr": bool,
     "ocr_applied": bool, "skew_angle": float | null, "blocks": [ {id, page, text, font_size, size_ratio, font_name,
               bold_ratio, italic_ratio, bbox, rel_x0, rel_y0, rel_y1,
               rel_width, n_chars, n_words, n_lines, digit_ratio, upper_ratio,
               ends_with_period, starts_with_bullet, looks_like_page_number,
               repeated_margin, source ("text" | "ocr"), ocr_conf, low_confidence} ]}
  ]
}
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00a0": " ", "\u200b": "", "\u00ad": "",
}
BOLD_FLAG, ITALIC_FLAG = 16, 2
BOLD_NAME_HINTS = ("bold", "black", "heavy", "semibold", "demibold", "demi", "extrabold", "ultra")
ITALIC_NAME_HINTS = ("italic", "oblique")
PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)
BULLET_RE = re.compile(r"^\s*([\u2022\u25cf\u25aa\-*]|\d+[.)])\s+")
MIN_DIGITAL_CHARS = 20  # a page with fewer characters AND images is treated as scanned


def normalize(text: str) -> str:
    """Replace ligatures / invisible characters."""
    for old, new in LIGATURES.items():
        text = text.replace(old, new)
    return text


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines and repair 'infor-' + 'mation' style hyphenation."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out += " " + line
    return re.sub(r"\s+", " ", out).strip()


def is_bold_font(font_name: str, flags: int = 0) -> bool:
    """Bold if the PDF flag is set OR the font name carries a weight keyword."""
    name = font_name.lower()
    return bool(flags & BOLD_FLAG) or any(h in name for h in BOLD_NAME_HINTS)


def is_italic_font(font_name: str, flags: int = 0) -> bool:
    name = font_name.lower()
    return bool(flags & ITALIC_FLAG) or any(h in name for h in ITALIC_NAME_HINTS)


def extract_page(page: "pymupdf.Page", page_no: int) -> dict:
    width, height = page.rect.width, page.rect.height
    data = page.get_text("dict", sort=True)
    blocks: list[dict] = []

    for b in data["blocks"]:
        if b["type"] != 0:
            continue
        lines: list[str] = []
        sizes: list[tuple[float, int]] = []
        fonts: Counter = Counter()
        total = bold = italic = 0

        for line in b["lines"]:
            parts = []
            for span in line["spans"]:
                text = normalize(span["text"])
                parts.append(text)
                n = len(text.strip())
                if n == 0:
                    continue
                total += n
                sizes.append((span["size"], n))
                fonts[span["font"]] += n
                if is_bold_font(span["font"], span["flags"]):
                    bold += n
                if is_italic_font(span["font"], span["flags"]):
                    italic += n
            lines.append("".join(parts))

        text = join_lines(lines)
        if not text or total == 0:
            continue

        x0, y0, x1, y1 = b["bbox"]
        alpha = [c for c in text if c.isalpha()]
        blocks.append({
            "id": f"p{page_no}_b{len(blocks)}",
            "page": page_no,
            "text": text,
            "font_size": round(sum(s * n for s, n in sizes) / total, 2),
            "font_name": fonts.most_common(1)[0][0],
            "bold_ratio": round(bold / total, 2),
            "italic_ratio": round(italic / total, 2),
            "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
            "rel_x0": round(x0 / width, 3),
            "rel_y0": round(y0 / height, 3),
            "rel_y1": round(y1 / height, 3),
            "rel_width": round((x1 - x0) / width, 3),
            "n_chars": len(text),
            "n_words": len(text.split()),
            "n_lines": len(lines),
            "digit_ratio": round(sum(c.isdigit() for c in text) / len(text), 3),
            "upper_ratio": round(sum(c.isupper() for c in alpha) / len(alpha), 3) if alpha else 0.0,
            "ends_with_period": text.rstrip().endswith((".", "!", "?")),
            "starts_with_bullet": bool(BULLET_RE.match(text)),
            "looks_like_page_number": bool(PAGE_NUM_RE.match(text)),
            "repeated_margin": False,
            "source": "text",
            "ocr_conf": None,
            "low_confidence": False,
        })

    n_chars = sum(b["n_chars"] for b in blocks)
    needs_ocr = n_chars < MIN_DIGITAL_CHARS and bool(page.get_images())
    return {
        "page": page_no,
        "width": width,
        "height": height,
        "needs_ocr": needs_ocr,
        "ocr_applied": False,
        "skew_angle": None,
        "blocks": blocks,
    }


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())[:80]


def flag_repeated_margins(pages: list[dict]) -> None:
    """Flag top/bottom-margin text that repeats on >= 40% of pages (headers/footers)."""
    if len(pages) < 3:
        return
    seen: dict[str, set[int]] = {}
    for p in pages:
        for b in p["blocks"]:
            if b["rel_y1"] < 0.12 or b["rel_y0"] > 0.88:
                seen.setdefault(_margin_key(b["text"]), set()).add(p["page"])
    repeated = {k for k, v in seen.items() if len(v) / len(pages) >= 0.4}
    for p in pages:
        for b in p["blocks"]:
            if (b["rel_y1"] < 0.12 or b["rel_y0"] > 0.88) and _margin_key(b["text"]) in repeated:
                b["repeated_margin"] = True


def add_size_ratios(pages: list[dict]) -> float:
    counter: Counter = Counter()
    for p in pages:
        for b in p["blocks"]:
            if not b["repeated_margin"] and not b["looks_like_page_number"]:
                counter[round(b["font_size"], 1)] += b["n_chars"]
    body = counter.most_common(1)[0][0] if counter else 12.0
    for p in pages:
        for b in p["blocks"]:
            b["size_ratio"] = round(b["font_size"] / body, 2)
    return body


def extract_document(pdf_path: str | Path, ocr: bool = True, dpi: int = 300,
                     lang: str = "eng", debug_dir: str | Path | None = None) -> dict:
    """Extract structured blocks. Scanned pages are OCR'd when ocr=True."""
    pdf_path = Path(pdf_path)
    ocr_pages: list[int] = []
    with pymupdf.open(pdf_path) as doc:
        pages = []
        for i, page in enumerate(doc):
            result = extract_page(page, i + 1)
            if result["needs_ocr"] and ocr:
                from src.ocr.engine import ocr_page
                blocks, info = ocr_page(page, i + 1, dpi=dpi, lang=lang, debug_dir=debug_dir)
                result["blocks"] = blocks
                result["ocr_applied"] = True
                result["skew_angle"] = info["skew_angle"]
                ocr_pages.append(i + 1)
            pages.append(result)
    flag_repeated_margins(pages)
    body = add_size_ratios(pages)
    return {
        "source": pdf_path.name,
        "num_pages": len(pages),
        "body_font_size": body,
        "ocr_pages": ocr_pages,
        "pages": pages,
    }


def save_json(result: dict, out_dir: str | Path = "data/processed") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(result["source"]).stem + ".json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured blocks from a PDF (offline).")
    ap.add_argument("pdf")
    ap.add_argument("--no-ocr", action="store_true", help="do not OCR scanned pages")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--debug", action="store_true", help="save preprocessed OCR images to data/processed/debug")
    args = ap.parse_args()

    result = extract_document(
        args.pdf,
        ocr=not args.no_ocr,
        dpi=args.dpi,
        lang=args.lang,
        debug_dir="data/processed/debug" if args.debug else None,
    )
    out = save_json(result)
    n_blocks = sum(len(p["blocks"]) for p in result["pages"])
    still_scanned = [p["page"] for p in result["pages"] if p["needs_ocr"] and not p["ocr_applied"]]
    print(f"Saved: {out}")
    print(f"Pages: {result['num_pages']} | Blocks: {n_blocks} | Body font: {result['body_font_size']}pt")
    print(f"OCR applied to pages: {result['ocr_pages'] or 'none'}")
    if still_scanned:
        print(f"WARNING pages still needing OCR (run without --no-ocr): {still_scanned}")
    for p in result["pages"]:
        if p["ocr_applied"]:
            confs = [b["ocr_conf"] for b in p["blocks"]]
            avg = round(sum(confs) / len(confs), 1) if confs else 0
            low = sum(b["low_confidence"] for b in p["blocks"])
            print(f"  page {p['page']}: skew={p['skew_angle']}deg, blocks={len(p['blocks'])}, avg_conf={avg}, low_conf_blocks={low}")
    first = result["pages"][0]["blocks"] if result["pages"] else []
    for b in first[:10]:
        print(f"  [{b['size_ratio']:.2f}x bold={b['bold_ratio']:.1f} {b['source']}] {b['text'][:70]}")


if __name__ == "__main__":
    main()
