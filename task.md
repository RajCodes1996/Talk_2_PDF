# task_day9.md: Question-type classifier and routing plan

Goal: `plan(question)` returns `{type, margin, page, handler}` so Day 10 can send each question to the right answer handler. Offline, scikit-learn only (TF-IDF word + char n-grams, LinearSVC). Seven types: definition, wh_fact (who/when/where/which), number, yes_no, summary, page_lookup, other. (`other` is added to the handover's six: why/how/compare questions would otherwise be forced into wrong types.)

## Rules (override AGENTS.md where they conflict)
- Work in `C:\Users\netware\Desktop\vision`, run from project root only.
- FIRST: `python -c "import sys; print(sys.version); print(sys.executable)"` must show **3.11.9** and a path inside `vision\.venv`. Otherwise STOP and report.
- Forbidden: uv/poetry/conda, recreating `.venv`, `pyproject.toml`/`uv.lock`, `import fitz`, new pip packages, any network/API/LLM call. Use `python -m pytest`, `python -m pip`.
- Create/edit ONLY the files named in this task. Never create, regenerate or overwrite any PDF in `data\samples`.
- Output discipline: pipe long output through `Select-Object -First/-Last N` as written. Full-suite runs are exempt from the "narrowest test" rule.
- Report REAL output. Do not write "all tests exist and pass" unless the per-file counts below prove it. "Deviations" must list every file touched or created beyond this task.
- If a step fails twice: stop, report the last 20 lines of the error plus 2 options.

## Part A: Preflight (read-only)
```
python -m pytest tests/test_retrieval.py tests/test_classifier.py -q | Select-Object -Last 3
git status --short | Select-Object -First 10
```
Expected: **12 passed** (9 + 3) after task_day8b.md; clean working tree. If not, STOP and report.

## Part B: Create files (`src/nlp/` and `tools/` already exist; create `data\questions` if missing)

### `src/nlp/question_classifier.py`
```python
"""Question-type classifier (TF-IDF word+char n-grams + LinearSVC) and page-number parsing."""
import csv
import re
from pathlib import Path

import joblib
import numpy as np

LABELS = ["definition", "wh_fact", "number", "yes_no", "summary", "page_lookup", "other"]
MODEL_PATH = Path("data/models/question_clf.joblib")
MIN_MARGIN = 0.1  # below this top1-top2 margin the type falls back to "other" (from heldout table)
# Day 10 handlers keyed by type (names only; handlers come on Day 10)
ROUTES = {"definition": "define", "wh_fact": "fact", "number": "number", "yes_no": "yes_no",
          "summary": "summarize", "page_lookup": "page", "other": "passages"}

_NUMW = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_ORD = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth".split())}


def normalize(q):
    return re.sub(r"\d+", " 0 ", q.lower().strip())


def make_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline as mk
    from sklearn.svm import LinearSVC
    feats = FeatureUnion([
        ("w", TfidfVectorizer(preprocessor=normalize, token_pattern=r"(?u)\b\w+\b",
                              ngram_range=(1, 2), sublinear_tf=True)),
        ("c", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                              ngram_range=(2, 5), sublinear_tf=True))])
    return mk(feats, LinearSVC(C=0.5, class_weight="balanced", random_state=42))


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] in LABELS and r["text"].strip()]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def train(train_csv, out_path=MODEL_PATH):
    X, y = load_csv(train_csv)
    m = make_pipeline().fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, out_path)
    return m


_cache = {}


def load_model(path=MODEL_PATH):
    if str(path) not in _cache:
        _cache[str(path)] = joblib.load(path)
    return _cache[str(path)]


def predict(question, model=None, min_margin=None):
    """-> (label, margin). Low margin falls back to 'other'."""
    m = model or load_model()
    mm = MIN_MARGIN if min_margin is None else min_margin
    s = m.decision_function([question])[0]
    order = np.argsort(-s)
    margin = float(s[order[0]] - s[order[1]])
    label = str(m.classes_[order[0]])
    return (label if margin >= mm else "other"), round(margin, 3)


def extract_page(q):
    """Page number named in a question: int, -1 for 'last page', None if none found."""
    t = q.lower()
    if re.search(r"\blast page\b|\bfinal page\b", t):
        return -1
    m = re.search(r"\bpage\s*(?:no\.?|number|#)?\s*(\d+)\b|\bp\.?\s*(\d+)\b", t)
    if m:
        return int(m.group(1) or m.group(2))
    m = re.search(r"\bpage\s+(" + "|".join(_NUMW) + r")\b", t)
    if m:
        return _NUMW[m.group(1)]
    m = re.search(r"\b(" + "|".join(_ORD) + r")\s+page\b", t)
    return _ORD[m.group(1)] if m else None


def plan(question, model=None, min_margin=None):
    label, margin = predict(question, model, min_margin)
    page = extract_page(question)
    if label == "page_lookup" and page is None:
        label = "other"  # no page named: cannot navigate
    return {"type": label, "margin": margin, "page": page, "handler": ROUTES[label]}


if __name__ == "__main__":
    import sys
    print(plan(" ".join(sys.argv[1:])))
```
### `tools/make_question_data.py`
```python
"""python -m tools.make_question_data  -> data/questions/train.csv and heldout.csv (deterministic).
train = templates x topics + hand-written; heldout = 70 separate hand-written questions (10 per type).
Labelling rules: number = asks for a quantity/amount/duration/ratio/count; definition = 'what is <concept>';
wh_fact = who/when/where/which/what year; yes_no = starts with an auxiliary verb; summary = overview of the
whole document or a section; page_lookup = names a page to go to/read; other = why/how/compare/explain/open."""
import csv
import random
from pathlib import Path

T = ["a screen reader", "a braille display", "contrast ratio", "speech rate", "OCR", "text-to-speech", "alt text",
     "heading structure", "photosynthesis", "inflation", "renewable energy", "a solar panel", "a carbon footprint",
     "vitamin D", "dietary fibre", "the Industrial Revolution", "compound interest", "a mortgage",
     "the budget deficit", "the Treaty of Versailles", "machine learning", "a neural network", "NVDA", "the keyboard shortcut",
     "a hash function", "the water cycle", "GDP", "a bond", "protein", "wind power"]
PEOPLE = ["the author", "the report", "the policy", "the study", "the guidelines", "the treaty", "the project", "the survey"]
EVENTS = ["the war end", "the law pass", "the merger happen", "the experiment take place", "the conference start",
          "prices rise", "the factory open", "the study begin"]
PLACES = ["the headquarters", "the lab", "the conference", "the factory", "the museum", "the signing", "the trial", "the office"]
SECTIONS = ["finance", "history", "nutrition", "energy", "accessibility", "methods", "results", "introduction", "keyboard navigation"]


def pick(rng, templates, n, slots):
    out = set()
    while len(out) < n:
        t = rng.choice(templates)
        out.add(t.format(**{k: rng.choice(v) for k, v in slots.items()}))
    return sorted(out)


def build_train(rng):
    d = {}
    d["definition"] = pick(rng, [
        "What is {t}?", "What is {t}", "Define {t}.", "What does {t} mean?", "Tell me what {t} is.", "What is meant by {t}?",
        "Give me a definition of {t}.", "Can you define {t}?", "What exactly is {t}?", "what's {t}", "Please define {t}",
        "What do you mean by {t}?"], 40, {"t": T}) + [
        "What does WCAG stand for?", "What does the abbreviation OCR stand for?", "Meaning of the word fibre?",
        "What is the definition of inflation?", "Define the term GDP", "What do they mean by heading levels?",
        "What is the meaning of alt text?", "What is a refreshable braille display?", "Explain the term compound interest."]
    d["wh_fact"] = pick(rng, [
        "Who wrote {p}?", "Who is responsible for {p}?", "Who founded the company behind {t}?", "Who invented {t}?",
        "When was {t} introduced?", "When did {e}?", "Where was {pl} located?", "Where is {pl}?", "Which company makes {t}?",
        "In what year was {t} first released?", "Which organisation published {p}?", "Who funded {p}?",
        "Where can I find {t}?", "When was {p} published?", "Who are the authors of {p}?", "Which country introduced {t}?",
        "What year did {e}?", "Where did {e}?"], 44, {"t": T, "p": PEOPLE, "e": EVENTS, "pl": PLACES}) + [
        "Who is the director?", "Name the author of this report.", "Which city hosted the event?", "Whom did they interview?",
        "When did it all start?", "Who is quoted in the introduction?"]
    d["number"] = pick(rng, [
        "How many words per minute do {w} prefer?", "How many {u} are there in {t}?", "How much does {t} cost?",
        "What percentage of {x} was reported?", "How long does {t} take?", "How many {u} does {p} mention?",
        "What is the recommended {m}?", "What was the total {m} in the report?", "How many people took part in {p}?",
        "By how much did {m} change?", "How high is the {m}?", "What is the average {m}?", "How many pages does {p} have?",
        "What is the ratio required for {t}?", "How large is the {m}?", "How big was the {m}?"], 44,
        {"w": ["beginners", "experts", "children", "most users"], "u": ["cells", "steps", "chapters", "examples", "countries", "tables"],
         "t": T, "p": PEOPLE, "x": ["adults", "students", "households", "voters", "patients"],
         "m": ["speech rate", "contrast ratio", "budget", "interest rate", "growth rate", "sample size", "price", "temperature"]}) + [
        "How many cells does a typical braille display have?", "What is the minimum contrast ratio for normal text?",
        "How much did the deficit grow in 2022?", "How many words per minute is considered fast?",
        "What was the inflation rate last year?", "How many hours of sunlight are needed?"]
    d["yes_no"] = pick(rng, [
        "Is {t} free?", "Does {t} support keyboard navigation?", "Are {t} and {u} the same thing?", "Can {t} be used offline?",
        "Should beginners use {t}?", "Is {t} mentioned in the document?", "Was {t} widely adopted?", "Do I need {t}?",
        "Has {t} changed recently?", "Will {t} work on Windows?", "Is {t} required by law?", "Did {p} mention {t}?",
        "Does {p} recommend {t}?", "Is {t} open source?", "Are there any limits on {t}?", "Could {t} be replaced?"], 44,
        {"t": T, "u": T, "p": PEOPLE}) + [
        "Is it free?", "Does it work offline?", "Is this document about finance?", "Are headings read aloud?",
        "Can I use it with a keyboard only?", "Was the policy approved?"]
    d["summary"] = pick(rng, [
        "Summarize this document.", "Give me a summary.", "What is this document about?", "What are the main points?",
        "Give me an overview of this document.", "Can you sum up the document?", "What is the gist of this?",
        "Tell me briefly what this is about.", "TL;DR", "Summarize the section on {s}.", "Give me a short summary of the part about {s}.",
        "What are the key takeaways?", "Overview please", "Briefly describe the contents of this file.", "Sum up the {s} section.",
        "What does the {s} section cover?", "Summarise the {s} part in two sentences.", "Recap the document for me."], 30,
        {"s": SECTIONS}) + ["What is this paper about?", "Give me the highlights.", "Main idea of the text?"]
    ords = ["first", "second", "third", "fourth", "fifth", "last", "tenth"]
    d["page_lookup"] = pick(rng, [
        "What is on page {n}?", "Read page {n}.", "Go to page {n}.", "Take me to page {n}.", "Summarize page {n}.",
        "What does page {n} say?", "Jump to page {n}.", "Read me page {n} again.", "Turn to page {n}.", "Show page {n}",
        "page {n} please", "Open page {n}", "Go back to page {n}.", "What is written on page {n}?", "Read the {o} page.",
        "Go to the {o} page.", "What is on the {o} page?", "Start reading from page {n}.", "Skip to page {n}.", "Read p. {n}"], 44,
        {"n": [str(i) for i in range(1, 41)] + ["two", "three", "five", "seven"], "o": ords}) + [
        "Go to the last page.", "Read the final page.", "Take me to the first page."]
    d["other"] = pick(rng, [
        "Why is {t} important?", "How does {t} work?", "Explain how {t} works.", "What are the advantages of {t}?",
        "Compare {t} and {u}.", "How can I use {t}?", "Why did {e}?", "What are the benefits of {t}?", "How do I set up {t}?",
        "What are the disadvantages of {t}?", "What is the difference between {t} and {u}?", "Tell me more about {t}.",
        "How is {t} different from {u}?", "Why would someone choose {t}?", "How should {t} be configured?", "Describe how {t} is used."],
        44, {"t": T, "u": T, "e": EVENTS}) + [
        "What can I do to reduce my carbon footprint?", "What happens if the contrast is too low?", "Any tips for beginners?",
        "What problems does the author raise?", "What does the author think about it?", "Can you help me understand this?"]
    return d


HELD = {
    "definition": ["What exactly is a screen reader?", "Could you tell me the meaning of alt text?", "What do they mean by compound interest?",
                   "Define the term carbon footprint.", "What's OCR?", "Explain the term inflation.", "What is NVDA?",
                   "What does GDP stand for?", "Meaning of dietary fibre?", "What is meant by renewable energy?"],
    "number": ["How many words per minute do experts listen at?", "What contrast ratio is recommended?", "How long does the battery last?",
               "What percentage of adults are affected?", "How much does it cost?", "How many cells does a braille display have?",
               "What is the speech rate for beginners?", "By how much did prices rise?", "How many pages does the report have?",
               "What was the total revenue in 2023?"],
    "wh_fact": ["Who is the author of this report?", "Which year did the treaty get signed?", "Where was the conference held?",
                "Who funds the project?", "When was NVDA released?", "Name the company that makes JAWS.",
                "Which organisation published these guidelines?", "Where do the authors work?", "What year did the war end?",
                "Who are the people mentioned in the history section?"],
    "yes_no": ["Is NVDA free to use?", "Does the document mention braille?", "Can I use it without internet?", "Are headings read aloud?",
               "Was the policy approved?", "Do beginners prefer slower speech?", "Is the report from 2023?",
               "Should large text have a lower contrast ratio?", "Has the budget increased?", "Will it work on Linux?"],
    "summary": ["Summarise the whole thing for me.", "What's this file about?", "Quick overview please.", "Give me the key points of the document.",
                "Tell me the main idea.", "Sum up chapter two.", "Briefly, what does it cover?", "What are the highlights?",
                "Short summary of the finance section.", "Can you recap what this text says?"],
    "page_lookup": ["Read page 12 to me.", "Take me to the third page.", "What's written on page 7?", "Open page 2.", "Go back to page 1.",
                    "Jump to the last page.", "Show me the second page.", "page five", "Start from page 9.", "What does page 4 contain?"],
    "other": ["Why does contrast matter for low vision users?", "How do screen readers handle images?", "Explain how OCR works.",
              "What are the pros and cons of braille displays?", "Compare NVDA and JAWS.", "How should I configure the voice speed?",
              "Why did inflation rise?", "What can I do about eye strain?", "Tell me more about the Industrial Revolution.",
              "What are the benefits of solar power?"],
}


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)


def make(out_dir="data/questions", seed=42):
    rng = random.Random(seed)
    tr = build_train(rng)
    hold_texts = {q.lower() for v in HELD.values() for q in v}
    train_rows = [(q, k) for k, v in tr.items() for q in v if q.lower() not in hold_texts]
    write(Path(out_dir) / "train.csv", train_rows)
    write(Path(out_dir) / "heldout.csv", [(q, k) for k, v in HELD.items() for q in v])
    return len(train_rows), sum(len(v) for v in HELD.values())


if __name__ == "__main__":
    print("train, heldout rows:", make())
```
### `tools/train_questions.py`
```python
"""python -m tools.train_questions  -> trains on data/questions/train.csv, evaluates on heldout.csv
(+ data/questions/heldout_user.csv if present), prints per-class report, confusion, margin table, errors.
Saves data/models/question_clf.joblib and data/models/question_log.txt"""
import numpy as np
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.nlp.question_classifier import LABELS, MODEL_PATH, load_csv, make_pipeline, train

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


Xtr, ytr = load_csv("data/questions/train.csv")
Xh, yh = load_csv("data/questions/heldout.csv")
if Path("data/questions/heldout_user.csv").exists():
    xu, yu = load_csv("data/questions/heldout_user.csv")
    Xh, yh = Xh + xu, yh + yu
    log(f"including {len(xu)} user-written heldout questions")
log(f"train={len(Xtr)} heldout={len(Xh)} counts={dict(Counter(ytr))}")

cv = cross_val_predict(make_pipeline(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42))
log(f"5-fold CV on train (optimistic, templated data): macro-F1={f1_score(ytr, cv, average='macro'):.3f}")

model = train("data/questions/train.csv", MODEL_PATH)
pred = list(model.predict(Xh))
log(f"HELDOUT macro-F1={f1_score(yh, pred, average='macro'):.3f} accuracy={np.mean(np.array(pred) == np.array(yh)):.3f}")
log(classification_report(yh, pred, labels=LABELS, zero_division=0))
cm = confusion_matrix(yh, pred, labels=LABELS)
log("confusion (rows=true, cols=pred): " + " ".join(l[:5] for l in LABELS))
for l, r in zip(LABELS, cm):
    log(f"{l[:9]:>9} " + " ".join(f"{v:5d}" for v in r))

S = model.decision_function(Xh)
srt = np.sort(S, axis=1)
margin = srt[:, -1] - srt[:, -2]
log("margin threshold -> coverage, accuracy on kept:")
for t in (0.0, 0.1, 0.2, 0.3, 0.5):
    keep = margin >= t
    acc = np.mean(np.array(pred)[keep] == np.array(yh)[keep]) if keep.any() else 0
    log(f"  {t}: coverage={keep.mean():.2f} acc={acc:.3f}")
log("misclassified (true -> pred | margin | text):")
for x, t, p, m in zip(Xh, yh, pred, margin):
    if t != p:
        log(f"  {t} -> {p} | {m:.2f} | {x}")
Path("data/models/question_log.txt").write_text("\n".join(lines), encoding="utf-8")
```
### `tests/test_questions.py`
```python
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
```

## Part C: Verify (agent)
```
python -m tools.make_question_data
python -m pytest tests/test_questions.py -q
python -m tools.train_questions
python -m src.nlp.question_classifier "Take me to page 9"
python -m src.nlp.question_classifier "Is NVDA free?"
python -m src.nlp.question_classifier "How many words per minute do beginners prefer?"
python -m src.nlp.question_classifier "Why does contrast matter?"
python -m pytest -q | Select-Object -Last 3
```
Expected: `train, heldout rows: (327, 70)`; **9 passed** (about 3 s); heldout macro-F1 about **0.94** (anything 0.90 to 0.97 is normal; the weakest class is `yes_no`); CLI types `page_lookup` (page 9), `yes_no`, `number`, `other`. Full suite: 28 + 9 = **37** while reader has 2 tests, **43** after the reader is restored. Report the full `train_questions` output (about 45 lines) and the four CLI lines.

## Part D: USER step (about 10 minutes, not the agent)
The built-in heldout set was written by Claude in the same style as the training templates, so its score is optimistic. Create `data\questions\heldout_user.csv` with header `text,label` and **at least 30 questions in your own words** (4 or 5 per type) about your own documents. Labels: `definition`, `wh_fact`, `number`, `yes_no`, `summary`, `page_lookup`, `other`. Rules: number = asks for a quantity, duration, ratio or count; definition = "what is <concept>"; wh_fact = who/when/where/which/what year; yes_no = starts with is/are/does/can/was...; summary = overview of the whole document or a section; page_lookup = names a page to go to or read; other = why/how/compare/explain. Include some messy, short or informal phrasings.

## Part E: Final score and margin check (agent, after Part D)
Run `python -m tools.train_questions` again (it picks up `heldout_user.csv` automatically). Report the full output. Margin rule: keep `MIN_MARGIN = 0.1` if accuracy at 0.1 in the table is at least 0.95; otherwise set it (one-line edit in `src/nlp/question_classifier.py`) to the smallest threshold in the table that has accuracy at least 0.95 and coverage at least 0.80. Make no other changes. Do not add or edit training questions to fix heldout errors.

## Part F: Commit
```
git add src/nlp/question_classifier.py tools/make_question_data.py tools/train_questions.py tests/test_questions.py data/questions data/models/question_log.txt
git commit -m "Day 9: question-type classifier and routing plan"
git log --oneline -3
```
No push.

## Acceptance criteria
1. Python 3.11.9 in `.venv`; no new packages.
2. `tests/test_questions.py`: 9 passed; full-suite total matches Part C (state which reader case applies).
3. Heldout macro-F1 reported for the built-in set and, after Part D, for built-in + user questions. Target (not a gate): >= 0.90 on the combined set; report honestly if lower.
4. The four CLI checks give the expected types, and "Take me to page 9" gives `page: 9`.
5. Commit made locally.

## Report back (paste verbatim, nothing more)
`sys.version` + executable | Part A | Part C outputs | Part E output (after Part D) | Part F `git log` | Deviations.

## Known limitations (accepted)
- Training data is templated plus hand-written: expect real-world accuracy below the heldout score. The user-written heldout set is the honest number.
- Ambiguous forms such as "Is the report from 2023?" (yes_no vs wh_fact) and "What is the speech rate?" (number vs definition) will sometimes be wrong; low-margin predictions fall back to `other`, which Day 10 answers with plain top passages.
- `extract_page` is regex-based: English only, understands "page 12", "p. 3", "page five", "third page", "last page".
- `plan()` only returns handler names; the handlers themselves are Day 10.