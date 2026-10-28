import json

from ner_eval import data, errors
from ner_eval.data import Sentence

GOLD = [
    Sentence(["Ann", "in", "Rome"], ["B-PER", "O", "B-LOC"]),
    Sentence(["New", "York", "Times"], ["B-ORG", "I-ORG", "I-ORG"]),
]


def test_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda ds, splits: {"test": GOLD})
    path = tmp_path / "conll-roberta-large-cleanconll-test.jsonl"
    preds = [["B-PER", "O", "B-ORG"], ["B-LOC", "I-LOC", "O"]]
    path.write_text("".join(json.dumps(p) + "\n" for p in preds))

    d = errors.errors(path)
    assert (d["method"], d["dataset"], d["split"], d["n"]) == (
        "roberta-large",
        "cleanconll",
        "test",
        2,
    )
    assert d["kinds"] == {"type": 1, "boundary": 1}
    assert d["trans"] == [["LOC", "ORG", 1]]
    assert [s["i"] for s in d["sents"]] == [0, 1]
    assert d["scores"]["PER"] == [1.0, 1.0, 1.0, 1]


def test_write_page(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load", lambda ds, splits: {"test": GOLD})
    path = tmp_path / "conll-bert-cleanconll-test.jsonl"
    path.write_text("".join(json.dumps(s.tags) + "\n" for s in GOLD))

    errors.write_page(path, tmp_path / "out.html")
    page = (tmp_path / "out.html").read_text()
    assert "__DATA__" not in page and '"method":"bert"' in page
