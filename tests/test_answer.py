import pytest

from src.nlp.answer import Answerer, cite, textrank
from src.nlp.question_classifier import train
from tests.test_retrieval import make_doc
from tools.make_question_data import make


@pytest.fixture(scope="module")
def ans(tmp_path_factory):
    d = tmp_path_factory.mktemp("a")
    make(d)
    return Answerer.from_doc(make_doc(), qmodel=train(d / "train.csv", d / "m.joblib"))


@pytest.mark.parametrize("q,phrase,page", [
    ("What contrast ratio is recommended?", "4.5", 3),
    ("How many cells does a braille display have?", "14 and 80", 4),
    ("What is NVDA?", "NVDA is a free", 1),
    ("Is NVDA free?", "free", 1)])
def test_answers_carry_citation(ans, q, phrase, page):
    r = ans.answer(q)
    assert r["found"] and phrase in r["answer"] and f"page {page}" in r["answer"] and r["sources"][0]["page"] == page


def test_yes_no_gives_no_verdict(ans):
    a = ans.answer("Is NVDA free?")["answer"]
    assert a.startswith("The closest text says") and not a.lower().startswith("yes")


def test_page_summary_and_abstain(ans):
    r = ans.answer("Read page 2.")
    assert r["action"] == {"goto_page": 2} and r["answer"] == "Going to page 2."
    assert "contrast" in ans.answer("What is on page 3?")["answer"].lower()
    assert ans.answer("Read page 9.")["answer"] == "This document has 4 pages."
    assert ans.answer("What is the capital of France?")["found"] is False


def test_summary_and_helpers(ans):
    r = ans.answer("Summarize this document.")
    assert r["answer"].startswith("Summary of the document") and len(r["sources"]) == 5
    sec = ans.answer("Summarize the section on braille displays.")
    assert "braille" in sec["answer"].lower() and len(sec["sources"]) <= 3
    assert cite([2]) == "page 2" and cite([3, 2, 5]) == "pages 2, 3 and 5"
    assert textrank(["a b c d e f", "a b c d e g", "x y z w v u", "a b c x y z"], 2) == sorted(textrank(["a b c d e f", "a b c d e g", "x y z w v u", "a b c x y z"], 2))

