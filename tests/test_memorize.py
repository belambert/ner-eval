from ner_eval.data import Sentence
from ner_eval.methods.memorize import Memorize


def test_memorize_longest_match():
    model = Memorize()
    model.fit(
        [
            Sentence(["New", "York", "City"], ["B-LOC", "I-LOC", "I-LOC"]),
            Sentence(["New", "York", "Times"], ["B-ORG", "I-ORG", "I-ORG"]),
            Sentence(["York"], ["B-PER"]),
        ]
    )
    assert model.predict([["the", "New", "York", "Times", "and", "York"]]) == [
        ["O", "B-ORG", "I-ORG", "I-ORG", "O", "B-PER"]
    ]


def test_memorize_save_load(tmp_path):
    model = Memorize()
    model.fit([Sentence(["New", "York"], ["B-LOC", "I-LOC"])])
    model.save(tmp_path / "m")
    loaded = Memorize.load(str(tmp_path / "m"))
    assert (loaded.lex, loaded.max_len) == (model.lex, model.max_len)
