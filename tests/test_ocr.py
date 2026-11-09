import cv2
import numpy as np
import pymupdf
import pytest

from src.ocr.engine import tesseract_available
from src.ocr.preprocess import binarize, estimate_skew, preprocess, rotate, to_gray
from src.ingestion.extractor import extract_document

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")


def make_text_pdf(path, pages=2):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), "Introduction", fontsize=24, fontname="hebo")
        page.insert_text((72, 150), "Screen readers speak the text on the screen aloud.", fontsize=12, fontname="helv")
        page.insert_text((72, 175), "Offline processing protects the privacy of users.", fontsize=12, fontname="helv")
        page.insert_text((72, 200), "Tesseract is a free and open source OCR engine.", fontsize=12, fontname="helv")
    doc.save(path)
    doc.close()


def make_scanned_pdf(src, dst, rotate_deg=0.0):
    """Rasterise each page of src and build an image‑only PDF (a fake 'scan')."""
    out = pymupdf.open()
    with pymupdf.open(src) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            if rotate_deg:
                img = rotate(img, rotate_deg, border=255)
            ok, png = cv2.imencode('.png', img)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=png.tobytes())
    out.save(dst)
    out.close()


def test_skew_estimation_recovers_rotation(tmp_path):
    src = tmp_path / "t.pdf"
    make_text_pdf(src, pages=1)
    with pymupdf.open(src) as doc:
        pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    skewed = rotate(img, 4.0, border=255)
    est = estimate_skew(skewed)
    assert abs(est - (-4.0)) < 1.0, est


def test_preprocess_returns_binary_image(tmp_path):
    src = tmp_path / "t.pdf"
    make_text_pdf(src, pages=1)
    with pymupdf.open(src) as doc:
        pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    binary, info = preprocess(img)
    assert binary.ndim == 2
    assert set(np.unique(binary)).issubset({0, 255})
    assert "skew_angle" in info

@needs_tesseract
def test_scanned_pdf_is_ocrd(tmp_path):
    src, scan = tmp_path / "t.pdf", tmp_path / "scan.pdf"
    make_text_pdf(src, pages=2)
    make_scanned_pdf(src, scan)
    result = extract_document(scan)
    assert result["ocr_pages"] == [1, 2]
    text = " ".join(b["text"] for b in result["pages"][0]["blocks"]).lower()
    assert "introduction" in text
    assert "screen readers" in text
    assert "privacy" in text
    for b in result["pages"][0]["blocks"]:
        assert b["source"] == "ocr"
        assert b["ocr_conf"] >= 30
        assert "size_ratio" in b

@needs_tesseract
def test_skewed_scan_still_readable(tmp_path):
    src, scan = tmp_path / "t.pdf", tmp_path / "scan.pdf"
    make_text_pdf(src, pages=1)
    make_scanned_pdf(src, scan, rotate_deg=3.0)
    result = extract_document(scan)
    page = result["pages"][0]
    assert page["ocr_applied"] and abs(page["skew_angle"]) >= 1.0
    text = " ".join(b["text"] for b in page["blocks"]).lower()
    assert "introduction" in text and "tesseract" in text

@needs_tesseract
def test_mixed_pdf_only_ocrs_scanned_pages(tmp_path):
    digital, scan, mixed = (
        tmp_path / "d.pdf",
        tmp_path / "s.pdf",
        tmp_path / "m.pdf",
    )
    make_text_pdf(digital, pages=1)
    make_scanned_pdf(digital, scan)
    out = pymupdf.open()
    for f in (digital, scan, digital):
        with pymupdf.open(f) as d:
            out.insert_pdf(d)
    out.save(mixed)
    out.close()
    result = extract_document(mixed)
    assert result["ocr_pages"] == [2]
    assert result["pages"][0]["blocks"][0]["source"] == "text"
    assert result["pages"][1]["blocks"][0]["source"] == "ocr"
    assert result["pages"][2]["blocks"][0]["source"] == "text"

def test_no_ocr_flag_leaves_scanned_pages_flagged(tmp_path):
    src, scan = tmp_path / "t.pdf", tmp_path / "scan.pdf"
    make_text_pdf(src, pages=1)
    make_scanned_pdf(src, scan)
    result = extract_document(scan, ocr=False)
    assert result["pages"][0]["needs_ocr"] is True
    assert result["pages"][0]["ocr_applied"] is False
    assert result["pages"][0]["blocks"] == []

