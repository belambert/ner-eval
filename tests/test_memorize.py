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
