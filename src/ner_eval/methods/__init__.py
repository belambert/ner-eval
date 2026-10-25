"""NER methods and their registry."""

from typing import Protocol

from ner_eval.data import Sentence
from ner_eval.methods.memorize import Memorize


class Tagger(Protocol):
    def fit(self, train: list[Sentence]) -> None: ...

    def predict(self, sents: list[list[str]]) -> list[list[str]]: ...


METHODS: dict[str, type[Tagger]] = {"memorize": Memorize}
