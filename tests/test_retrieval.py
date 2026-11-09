import csv
import json

import pytest

from src.retrieval.chunker import build_chunks, split_sentences
from src.retrieval.index import Index, tokenize


def _b(page, n, text, size=1.0, **kw):
    d = {"id": f"p{page}_b{n}", "page": page, "text": text, "size_ratio": size, "n_words": len(text.split()),
         "n_lines": 1, "ends_with_period": text.endswith("."), "looks_like_page_number": False,
         "repeated_margin": False}
    d.update(kw)
    return d


def make_doc():
    P = [
        [_b(1, 0, "Screen Readers", 1.6),
         _b(1, 1, "NVDA is a free and open source screen reader for Windows. It converts text on the screen "
                  "into speech or braille output. Many blind users rely on it every day for web browsing and "
                  "document reading across applications."),
         _b(1, 2, "Accessibility Guide", .8, repeated_margin=True), _b(1, 3, "1", .8, looks_like_page_number=True)],
        [_b(2, 0, "Speech Rate", 1.6),
         _b(2, 1, "Beginners usually prefer a slower speech rate of about 150 words per minute. Experienced users "
                  "often listen at more than 300 words per minute without losing comprehension of the content."),
         _b(2, 2, "Accessibility Guide", .8, repeated_margin=True), _b(2, 3, "2", .8, looks_like_page_number=True)],
        [_b(3, 0, "Visual Contrast", 1.6),
         _b(3, 1, "Text should have a contrast ratio of at least 4.5 to 1 against its background. Large text "
                  "needs a ratio of at least 3 to 1 under the common accessibility guidelines for the web."),
         _b(3, 2, "Accessibility Guide", .8, repeated_margin=True), _b(3, 3, "3", .8, looks_like_page_number=True)],
        [_b(4, 0, "Braille Displays", 1.6),
         _b(4, 1, "A refreshable braille display raises and lowers small pins to form braille cells. Devices "
                  "usually offer between 14 and 80 cells and connect over USB or Bluetooth."),
         _b(4, 2, "Accessibility Guide", .8, repeated_margin=True), _b(4, 3, "4", .8, looks_like_page_number=True)],
    ]
    return {"source": "t.pdf", "num_pages": 4, "pages": [{"page": i + 1, "blocks": b} for i, b in enumerate(P)]}


def test_chunks_skip_margins_and_keep_page_metadata():
    ch = build_chunks(make_doc())
    assert [c["heading"] for c in ch] == ["Screen Readers", "Speech Rate", "Visual Contrast", "Braille Displays"]
    assert [c["page_start"] for c in ch] == [1, 2, 3, 4]
    assert all("Accessibility Guide" not in c["text"] for c in ch)
    assert all(c["text"] not in ("1", "2", "3", "4") for c in ch)


def test_labels_override_heuristics_and_long_block_splits():
    doc = {"pages": [{"page": 1, "blocks": [
        _b(1, 0, "Intro", 1.0, label="heading"),
        _b(1, 1, " ".join(f"Sentence number {i} has exactly seven words." for i in range(30)), label="paragraph"),
        _b(1, 2, "Footer text here", label="footer")]}]}
    ch = build_chunks(doc, max_words=60, min_words=15)
    assert len(ch) > 3 and all(c["heading"] == "Intro" for c in ch)
    assert all(c["n_words"] <= 60 + 8 for c in ch) and "Footer" not in " ".join(c["text"] for c in ch)


def test_split_sentences_and_tokenize():
    assert len(split_sentences("One two. Three four! Five six?")) == 3
    assert tokenize("The 4.5 ratios, NVDA's reading") == ["4.5", "ratio", "nvda", "read"]


@pytest.mark.parametrize("q,page", [
    ("Is NVDA free?", 1), ("What speed do beginners prefer?", 2),
    ("What contrast ratio is needed?", 3), ("How many cells does a braille display have?", 4)])
def test_query_finds_right_page(q, page):
    idx = Index.from_doc(make_doc())
    res = idx.query(q, k=3)
    assert res and res[0]["page"] == page and 0 < res[0]["score"] <= 1


def test_edge_cases_and_alpha_modes_and_roundtrip(tmp_path):
    idx = Index.from_doc(make_doc())
    assert idx.query("the of and") == [] and idx.query("zzzqqq") == []
    for a in (0.0, 1.0):
        assert idx.query("braille pins", 1, alpha=a)[0]["page"] == 4
    f = tmp_path / "i.joblib"
    idx.save(f)
    assert Index.load(f).query("Is NVDA free?")[0]["page"] == 1
    with pytest.raises(ValueError):
        Index([])

