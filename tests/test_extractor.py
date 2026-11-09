import pymupdf

from src.ingestion.extractor import extract_document, is_bold_font, join_lines, normalize


def make_pdf(path):
    doc = pymupdf.open()
    for i in range(4):
        page = doc.new_page()  # A4: 595 x 842
        page.insert_text((72, 40), "Annual Report 2026", fontsize=9, fontname="helv")  # repeated header
        page.insert_text((72, 100), "Introduction", fontsize=22, fontname="hebo")       # bold heading
        page.insert_text((72, 140), "This is a normal paragraph of body text.", fontsize=11, fontname="helv")
        page.insert_text((290, 800), str(i + 1), fontsize=9, fontname="helv")           # page number
    doc.save(path)
    doc.close()


def test_structure(tmp_path):
    pdf = tmp_path / "t.pdf"
    make_pdf(pdf)
    result = extract_document(pdf)
    assert result["num_pages"] == 4

    blocks = result["pages"][0]["blocks"]
    heading = next(b for b in blocks if b["text"] == "Introduction")
    body = next(b for b in blocks if b["text"].startswith("This is"))

    assert heading["size_ratio"] > 1.5
    assert heading["bold_ratio"] == 1.0
    assert body["bold_ratio"] == 0.0
    assert any(b["looks_like_page_number"] for b in blocks)
    assert any(b["repeated_margin"] and "Annual" in b["text"] for b in blocks)
    assert all(b["source"] == "text" for b in blocks)
    assert result["ocr_pages"] == []


def test_cleaning_helpers():
    assert normalize("o\ufb03ce \ufb01le") == "office file"
    assert join_lines(["infor-", "mation is key"]) == "information is key"
    assert join_lines(["well", "known"]) == "well known"


def test_bold_font_detection():
    assert is_bold_font("Helvetica-Bold")
    assert is_bold_font("Montserrat-SemiBold")
    assert is_bold_font("Inter-Black")
    assert is_bold_font("SomeFont", flags=16)
    assert not is_bold_font("Helvetica")
    assert not is_bold_font("TimesNewRomanPSMT")
