"""Generate deterministic training and heldout question datasets.

Running as a module creates `data/questions/train.csv` and `data/questions/heldout.csv`.
"""

import csv
import random
from pathlib import Path

# Template vocabularies
T = [
    "a screen reader",
    "a braille display",
    "contrast ratio",
    "speech rate",
    "OCR",
    "text-to-speech",
    "alt text",
    "heading structure",
    "photosynthesis",
    "inflation",
    "renewable energy",
    "a solar panel",
    "a carbon footprint",
    "vitamin D",
    "dietary fibre",
    "the Industrial Revolution",
    "compound interest",
    "a mortgage",
    "the budget deficit",
    "the Treaty of Versailles",
    "machine learning",
    "a neural network",
    "NVDA",
    "the keyboard shortcut",
    "a hash function",
    "the water cycle",
    "GDP",
    "a bond",
    "protein",
    "wind power",
]

PEOPLE = ["the author", "the report", "the policy", "the study", "the guidelines", "the treaty", "the project", "the survey"]
EVENTS = ["the war end", "the law pass", "the merger happen", "the experiment take place", "the conference start", "prices rise", "the factory open", "the study begin"]
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
    # definition
    d["definition"] = pick(rng, [
        "What is {t}?",
        "What is {t}",
        "Define {t}.",
        "What does {t} mean?",
        "Tell me what {t} is.",
        "What is meant by {t}?",
        "Give me a definition of {t}.",
        "Can you define {t}?",
        "What exactly is {t}?",
        "what's {t}",
        "Please define {t}",
        "What do you mean by {t}?",
    ], 40, {"t": T}) + [
        "What does WCAG stand for?",
        "What does the abbreviation OCR stand for?",
        "Meaning of the word fibre?",
        "What is the definition of inflation?",
        "Define the term GDP",
        "What do they mean by heading levels?",
        "What is the meaning of alt text?",
        "What is a refreshable braille display?",
        "Explain the term compound interest.",
    ]
    # wh_fact
    d["wh_fact"] = pick(rng, [
        "Who wrote {p}?",
        "Who is responsible for {p}?",
        "Who founded the company behind {t}?",
        "Who invented {t}?",
        "When was {t} introduced?",
        "When did {e}?",
        "Where was {pl} located?",
        "Where is {pl}?",
        "Which company makes {t}?",
        "In what year was {t} first released?",
        "Which organisation published {p}?",
        "Who funded {p}?",
        "Where can I find {t}?",
        "When was {p} published?",
        "Who are the authors of {p}?",
        "Which country introduced {t}?",
        "What year did {e}?",
        "Where did {e}?",
    ], 44, {"t": T, "p": PEOPLE, "e": EVENTS, "pl": PLACES}) + [
        "Who is the director?",
        "Name the author of this report.",
        "Which city hosted the event?",
        "Whom did they interview?",
        "When did it all start?",
        "Who is quoted in the introduction?",
    ]
    # number
    d["number"] = pick(rng, [
        "How many words per minute do {w} prefer?",
        "How many {u} are there in {t}?",
        "How much does {t} cost?",
        "What percentage of {x} was reported?",
        "How long does {t} take?",
        "How many {u} does {p} mention?",
        "What is the recommended {m}?",
        "What was the total {m} in the report?",
        "How many people took part in {p}?",
        "By how much did {m} change?",
        "How high is the {m}?",
        "What is the average {m}?",
        "How many pages does {p} have?",
        "What is the ratio required for {t}?",
        "How large is the {m}?",
        "How big was the {m}?",
    ], 44, {
        "w": ["beginners", "experts", "children", "most users"],
        "u": ["cells", "steps", "chapters", "examples", "countries", "tables"],
        "t": T,
        "p": PEOPLE,
        "x": ["adults", "students", "households", "voters", "patients"],
        "m": ["speech rate", "contrast ratio", "budget", "interest rate", "growth rate", "sample size", "price", "temperature"],
    }) + [
        "How many cells does a typical braille display have?",
        "What is the minimum contrast ratio for normal text?",
        "How much did the deficit grow in 2022?",
        "How many words per minute is considered fast?",
        "What was the inflation rate last year?",
        "How many hours of sunlight are needed?",
    ]
    # yes_no
    d["yes_no"] = pick(rng, [
        "Is {t} free?",
        "Does {t} support keyboard navigation?",
        "Are {t} and {u} the same thing?",
        "Can {t} be used offline?",
        "Should beginners use {t}?",
        "Is {t} mentioned in the document?",
        "Was {t} widely adopted?",
        "Do I need {t}?",
        "Has {t} changed recently?",
        "Will {t} work on Windows?",
        "Is {t} required by law?",
        "Did {p} mention {t}?",
        "Does {p} recommend {t}?",
        "Is {t} open source?",
        "Are there any limits on {t}?",
        "Could {t} be replaced?",
    ], 44, {"t": T, "u": T, "p": PEOPLE}) + [
        "Is it free?",
        "Does it work offline?",
        "Is this document about finance?",
        "Are headings read aloud?",
        "Can I use it with a keyboard only?",
        "Was the policy approved?",
    ]
    # summary
    d["summary"] = pick(rng, [
        "Summarize this document.",
        "Give me a summary.",
        "What is this document about?",
        "What are the main points?",
        "Give me an overview of this document.",
        "Can you sum up the document?",
        "What is the gist of this?",
        "Tell me briefly what this is about.",
        "TL;DR",
        "Summarize the section on {s}.",
        "Give me a short summary of the part about {s}.",
        "What are the key takeaways?",
        "Overview please",
        "Briefly describe the contents of this file.",
        "Sum up the {s} section.",
        "What does the {s} section cover?",
        "Summarise the {s} part in two sentences.",
        "Recap the document for me.",
    ], 30, {"s": SECTIONS}) + [
        "What is this paper about?",
        "Give the highlights.",
        "Main idea of the text?",
    ]
    # page_lookup
    ords = ["first", "second", "third", "fourth", "fifth", "last", "tenth"]
    d["page_lookup"] = pick(rng, [
        "What is on page {n}?",
        "Read page {n}.",
        "Go to page {n}.",
        "Take me to page {n}.",
        "Summarize page {n}.",
        "What does page {n} say?",
        "Jump to page {n}.",
        "Read me page {n} again.",
        "Turn to page {n}.",
        "Show page {n}",
        "page {n} please",
        "Open page {n}",
        "Go back to page {n}.",
        "What is written on page {n}?",
        "Read the {o} page.",
        "Go to the {o} page.",
        "What is on the {o} page?",
        "Start reading from page {n}.",
        "Skip to page {n}.",
        "Read p. {n}",
    ], 44, {"n": [str(i) for i in range(1, 41)] + ["two", "three", "five", "seven"], "o": ords}) + [
        "Go to the last page.",
        "Read the final page.",
        "Take me to the first page.",
    ]
    # other
    d["other"] = pick(rng, [
        "Why is {t} important?",
        "How does {t} work?",
        "Explain how {t} works.",
        "What are the advantages of {t}?",
        "Compare {t} and {u}.",
        "How can I use {t}?",
        "Why did {e}?",
        "What are the benefits of {t}?",
        "How do I set up {t}?",
        "What are the disadvantages of {t}?",
        "What is the difference between {t} and {u}?",
        "Tell me more about {t}.",
        "How is {t} different from {u}?",
        "Why would someone choose {t}?",
        "How should {t} be configured?",
        "Describe how {t} is used.",
    ], 44, {"t": T, "u": T, "e": EVENTS}) + [
        "What can I do to reduce my carbon footprint?",
        "What happens if the contrast is too low?",
        "Any tips for beginners?",
        "What problems does the author raise?",
        "What does the author think about it?",
        "Can you help me understand this?",
    ]
    return d

HELD = {
    "definition": [
        "What exactly is a screen reader?",
        "Could you tell me the meaning of alt text?",
        "What do they mean by compound interest?",
        "Define the term carbon footprint.",
        "What's OCR?",
        "Explain the term inflation.",
        "What is NVDA?",
        "What does GDP stand for?",
        "Meaning of dietary fibre?",
        "What is meant by renewable energy?",
    ],
    "number": [
        "How many words per minute do experts listen at?",
        "What contrast ratio is recommended?",
        "How long does the battery last?",
        "What percentage of adults are affected?",
        "How much does it cost?",
        "How many cells does a braille display have?",
        "What is the speech rate for beginners?",
        "By how much did prices rise?",
        "How many pages does the report have?",
        "What was the total revenue in 2023?",
    ],
    "wh_fact": [
        "Who is the author of this report?",
        "Which year did the treaty get signed?",
        "Where was the conference held?",
        "Who funds the project?",
        "When was NVDA released?",
        "Name the company that makes JAWS.",
        "Which organisation published these guidelines?",
        "Where do the authors work?",
        "What year did the war end?",
        "Who are the people mentioned in the history section?",
    ],
    "yes_no": [
        "Is NVDA free to use?",
        "Does the document mention braille?",
        "Can I use it without internet?",
        "Are headings read aloud?",
        "Was the policy approved?",
        "Do beginners prefer slower speech?",
        "Is the report from 2023?",
        "Should large text have a lower contrast ratio?",
        "Has the budget increased?",
        "Will it work on Linux?",
    ],
    "summary": [
        "Summarise the whole thing for me.",
        "What's this file about?",
        "Quick overview please.",
        "Give me the key points of the document.",
        "Tell me the main idea.",
        "Sum up chapter two.",
        "Briefly, what does it cover?",
        "What are the highlights?",
        "Short summary of the finance section.",
        "Can you recap what this text says?",
    ],
    "page_lookup": [
        "Read page 12 to me.",
        "Take me to the third page.",
        "What's written on page 7?",
        "Open page 2.",
        "Go back to page 1.",
        "Jump to the last page.",
        "Show me the second page.",
        "page five",
        "Start from page 9.",
        "What does page 4 contain?",
    ],
    "other": [
        "Why does contrast matter for low vision users?",
        "How do screen readers handle images?",
        "Explain how OCR works.",
        "What are the pros and cons of braille displays?",
        "Compare NVDA and JAWS.",
        "How should I configure the voice speed?",
        "Why did inflation rise?",
        "What can I do about eye strain?",
        "Tell me more about the Industrial Revolution.",
        "What are the benefits of solar power?",
    ],
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
    train_rows = [(q, k) for k, v in tr.items() for q in v]
    write(Path(out_dir) / "train.csv", train_rows)
    write(Path(out_dir) / "heldout.csv", [(q, k) for k, v in HELD.items() for q in v])
    return len(train_rows), sum(len(v) for v in HELD.values())

if __name__ == "__main__":
    print("train, heldout rows:", make())

