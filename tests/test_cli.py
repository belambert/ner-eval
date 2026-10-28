import json

from typer.testing import CliRunner

from ner_eval import cli, data
from ner_eval.data import Sentence

SPLITS = {
    "train": [Sentence(["New", "York", "City"], ["B-LOC", "I-LOC", "I-LOC"])],
    "dev": [],
    "test": [
        Sentence(["New", "York", "City", "Hall"], ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]),
        Sentence(["in", "New", "York", "City"], ["O", "B-LOC", "I-LOC", "I-LOC"]),
        Sentence(["York"], ["B-LOC"]),
    ],
}


PREDS = [["B-LOC", "I-LOC", "I-LOC", "O"], ["O", "B-LOC", "I-LOC", "I-LOC"]]


class Fixed:
    """Tagger that returns PREDS, so CLI tests don't train a model."""

    def fit(self, train):
        pass

    def predict(self, sents):
        return PREDS[: len(sents)]


def run(*args):
    res = CliRunner().invoke(cli.app, [*map(str, args), "--json"])
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


def counts(row):
    return row["tp"], row["fp"], row["fn"]


def test_preds_rescore_round_trip(tmp_path, monkeypatch):
    monkeypatch.setitem(cli.METHODS, "fixed", Fixed)
    monkeypatch.setattr(
        data, "load", lambda ds, splits=data.SPLITS: {s: SPLITS[s] for s in splits}
    )
    [ev] = run("evaluate", "fixed", "cleanconll", "--max-eval", 2, "--preds", tmp_path)
    path = tmp_path / "conll-fixed-cleanconll-test.jsonl"
    assert path.read_text().splitlines() == [
        '["B-LOC","I-LOC","I-LOC","O"]',
        '["O","B-LOC","I-LOC","I-LOC"]',
    ]

    [rs] = run("rescore", path)
    assert (rs["dataset"], rs["split"], rs["n"]) == ("cleanconll", "test", 2)
    assert counts(rs) == counts(ev) == (1, 1, 1)
    [rs] = run("rescore", path, "--overlap", 0.5)
    assert counts(rs) == (2, 0, 0)
