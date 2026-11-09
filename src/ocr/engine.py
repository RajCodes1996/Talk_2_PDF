from __future__ import annotations

import os
import re
import shutil
import statistics
from pathlib import Path

import pymupdf
import pytesseract
from pytesseract import Output

from .preprocess import DEFAULT_DPI, preprocess, render_page

WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
MIN_WORD_CONF = 30        # words below this confidence are dropped
LOW_CONF_BLOCK = 60       # blocks below this mean confidence get low_confidence=True
PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)
BULLET_RE = re.compile(r"^\s*([\u2022\u25cf\u25aa\-*]|\d+[.)])\s+")


class TesseractNotFound(RuntimeError):
    pass


def configure_tesseract() -> str:
    """Locate tesseract (TESSERACT_CMD env var, then PATH, then default Windows folders)."""
    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), *WINDOWS_PATHS]
    for c in candidates:
        if c and Path(c).exists():
            pytesseract.pytesseract.tesseract_cmd = c
            return c
    raise TesseractNotFound(
        "Tesseract not found. Install it and add it to PATH, or set the TESSERACT_CMD "
        "environment variable to the full path of tesseract.exe."
    )


def tesseract_available() -> bool:
    try:
        configure_tesseract()
        return True
    except TesseractNotFound:
        return False


def _join_lines(lines: list[str]) -> str:
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


def ocr_image(binary, page_no: int, page_w_pt: float, page_h_pt: float, dpi: int,
              lang: str = "eng") -> list[dict]:
    """Run Tesseract on a preprocessed image and return blocks (paragraph level)."""
    configure_tesseract()
    data = pytesseract.image_to_data(binary, lang=lang, config="--oem 1 --psm 3",
                                     output_type=Output.DICT)
    px_to_pt = 72.0 / dpi
    paras: dict[tuple[int, int], dict] = {}

    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < MIN_WORD_CONF:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        p = paras.setdefault(key, {"lines": {}, "confs": [], "boxes": [], "line_h": {}})
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        p["lines"].setdefault(data["line_num"][i], []).append(text)
        p["confs"].append(conf)
        p["boxes"].append((x, y, x + w, y + h))
        ln = data["line_num"][i]
        p["line_h"][ln] = max(p["line_h"].get(ln, 0), h)

    blocks: list[dict] = []
    for key in sorted(paras):
        p = paras[key]
        lines = [" ".join(words) for _, words in sorted(p["lines"].items())]
        text = _join_lines(lines)
        if not text:
            continue
        x0 = min(b[0] for b in p["boxes"]) * px_to_pt
        y0 = min(b[1] for b in p["boxes"]) * px_to_pt
        x1 = max(b[2] for b in p["boxes"]) * px_to_pt
        y1 = max(b[3] for b in p["boxes"]) * px_to_pt
        font_size = float(max(6, round(statistics.median(p["line_h"].values()) * px_to_pt)))
        mean_conf = sum(p["confs"]) / len(p["confs"]) if p["confs"] else 0.0
        alpha = [c for c in text if c.isalpha()]
        blocks.append({
            "id": f"p{page_no}_b{len(blocks)}",
            "page": page_no,
            "text": text,
            "font_size": font_size,
            "font_name": "ocr",
            "bold_ratio": 0.0,
            "italic_ratio": 0.0,
            "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
            "rel_x0": round(x0 / page_w_pt, 3),
            "rel_y0": round(y0 / page_h_pt, 3),
            "rel_y1": round(y1 / page_h_pt, 3),
            "rel_width": round((x1 - x0) / page_w_pt, 3),
            "n_chars": len(text),
            "n_words": len(text.split()),
            "n_lines": len(lines),
            "digit_ratio": round(sum(c.isdigit() for c in text) / len(text), 3),
            "upper_ratio": round(sum(c.isupper() for c in alpha) / len(alpha), 3) if alpha else 0.0,
            "ends_with_period": text.rstrip().endswith((".", "!", "?")),
            "starts_with_bullet": bool(BULLET_RE.match(text)),
            "looks_like_page_number": bool(PAGE_NUM_RE.match(text)),
            "repeated_margin": False,
            "source": "ocr",
            "ocr_conf": round(mean_conf, 1),
            "low_confidence": mean_conf < LOW_CONF_BLOCK,
        })
    return blocks


def ocr_page(page: "pymupdf.Page", page_no: int, dpi: int = DEFAULT_DPI,
              lang: str = "eng", debug_dir: str | Path | None = None) -> tuple[list[dict], dict]:
    """Render → preprocess → OCR one PDF page. Returns (blocks, info)."""
    img = render_page(page, dpi)
    binary, info = preprocess(img)
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(debug_dir) / f"page_{page_no:03d}_ocr_input.png"), binary)
    blocks = ocr_image(binary, page_no, page.rect.width, page.rect.height, dpi, lang)
    info["n_blocks"] = len(blocks)
    return blocks, info

