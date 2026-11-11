import pytest
from sklearn.metrics import f1_score

from src.nlp.question_classifier import extract_page, load_csv, plan, train
from tools.make_question_data import make


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("q")
    make(d)
    m = train(d / "train.csv", d / "m.joblib")
    return m, d


def test_heldout_macro_f1(model):
    m, d = model
    X, y = load_csv(d / "heldout.csv")
    assert f1_score(y, m.predict(X), average="macro") >= 0.85


@pytest.mark.parametrize("q,t", [
    ("What is OCR?", "definition"), ("Who wrote the report?", "wh_fact"), ("How many words per minute do beginners prefer?", "number"),
    ("Is NVDA free?", "yes_no"), ("Summarize this document.", "summary"), ("Read page 4.", "page_lookup"),
    ("Why is contrast important?", "other")])
def test_plan_types(model, q, t):
    assert plan(q, model[0])["type"] == t


def test_extract_page_and_fallback(model):
    cases = {"Read page 12": 12, "go to p. 3": 3, "What is on page five?": 5, "the third page please": 3,
             "Jump to the last page": -1, "Is NVDA free?": None}
    for q, n in cases.items():
        assert extract_page(q) == n
    p = plan("Read page 4.", model[0])
    assert p["page"] == 4 and p["handler"] == "page" and set(p) == {"type", "margin", "page", "handler"}
    assert plan("Take me somewhere", model[0], min_margin=0)["type"] != "page_lookup"

