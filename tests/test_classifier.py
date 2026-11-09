import csv

import numpy as np
import pytest

from src.nlp.block_classifier import FEATURES, LABELS, classify_document, predict, train


def _synth(path, n_docs=4, per=42, seed=0):
    rng = np.random.default_rng(seed)
    protos = {  # size_ratio, bold, rel_y0, n_words, bullet, pagenum, repeated
        "heading": (1.6, .9, .2, 4, 0, 0, 0), "paragraph": (1.0, 0, .5, 60, 0, 0, 0),
        "list": (1.0, 0, .5, 12, 1, 0, 0), "table": (.95, 0, .5, 5, 0, 0, 0),
        "footer": (.8, 0, .95, 6, 0, 0, 1), "page_number": (.8, 0, .97, 1, 0, 1, 1),
        "caption": (.9, 0, .6, 10, 0, 0, 0)}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["doc", "block_id", "label", "how", "text"] + FEATURES)
        w.writeheader()
        for d in range(n_docs):
            for i in range(per):
                lab = LABELS[i % 7]
                s, b, y0, nw, bu, pn, rp = protos[lab]
                r = dict.fromkeys(FEATURES, 0)
                r.update(size_ratio=s + rng.normal(0, .03), bold_ratio=b,
                         rel_y0=y0 + rng.normal(0, .02), rel_y1=y0 + .03,
                         n_words=nw + int(rng.integers(0, 3)), n_chars=(nw + 2) * 6,
                         starts_with_bullet=bu, looks_like_page_number=pn, repeated_margin=rp,
                         rel_x0=.1, rel_width=.8, n_lines=1, is_ocr=0)
                w.writerow({"doc": f"d{d}", "block_id": f"p1_b{i}", "label": lab,
                            "how": "gold", "text": "x", **r})


def test_train_predict_and_doc(tmp_path):
    c, m = tmp_path / "l.csv", tmp_path / "m.joblib"
    _synth(c)
    res = train(c, m, log=lambda s: None)
    assert res["macro_f1"] > 0.8
    block = {"size_ratio": 1.6, "bold_ratio": .9, "rel_y0": .2, "n_words": 4, "n_chars": 36,
             "rel_y1": .23, "rel_x0": .1, "rel_width": .8, "n_lines": 1, "source": "text"}
    out = predict([dict(block)], m)
    assert out[0]["label"] == "heading" and 0 < out[0]["label_conf"] <= 1
    doc = classify_document({"pages": [{"blocks": [dict(block), dict(block)]}]}, m)
    assert all("label" in b for b in doc["pages"][0]["blocks"])


def test_needs_two_docs(tmp_path):
    c = tmp_path / "l.csv"
    _synth(c, n_docs=1)
    with pytest.raises(ValueError):
        train(c, tmp_path / "m.joblib", log=lambda s: None)

