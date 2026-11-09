from tools.label import COLS, FEATURES, gold_label, suggest, to_row


def blk(text, **kw):
    b = {
        "id": "p1_b0",
        "page": 1,
        "text": text,
        "size_ratio": 1.0,
        "bold_ratio": 0.0,
        "italic_ratio": 0.0,
        "rel_x0": 0.1,
        "rel_y0": 0.3,
        "rel_y1": 0.35,
        "rel_width": 0.8,
        "n_chars": len(text),
        "n_words": len(text.split()),
        "n_lines": 1,
        "digit_ratio": 0.0,
        "upper_ratio": 0.1,
        "ends_with_period": text.endswith('.'),
        "starts_with_bullet": False,
        "looks_like_page_number": False,
        "repeated_margin": False,
    }
    b.update(kw)
    return b


def test_suggest():
    assert suggest(blk("3", looks_like_page_number=True)) == ("page_number", True)
    assert suggest(blk("Acme Corp", repeated_margin=True)) == ("footer", True)
    assert suggest(blk("Figure 2: A chart")) == ("caption", False)
    assert suggest(blk("\u2022 item", starts_with_bullet=True)) == ("list", False)
    assert suggest(blk("Introduction", size_ratio=1.8, bold_ratio=1.0)) == ("heading", False)
    assert suggest(blk("A normal sentence of body text."))[0] == "paragraph"


def test_gold_matching():
    gold = {"first point about energy": "list", "a long paragraph sentence that continues here.": "paragraph"}
    assert gold_label(blk("First point about energy"), gold) == "list"
    assert gold_label(blk("\u2022 A long paragraph sentence that continues here."), gold) == "paragraph"
    assert gold_label(blk("Unknown text"), gold) is None


def test_row_has_all_columns():
    row = to_row("d.pdf", blk("Hello"), "paragraph", "manual")
    assert list(row) == COLS and len(FEATURES) == 17 and row["is_ocr"] == 0

