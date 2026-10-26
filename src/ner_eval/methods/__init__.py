"""NER methods and their registry."""

from collections.abc import Callable
from functools import partial
from typing import Protocol

from ner_eval.data import Sentence
from ner_eval.methods.memorize import Memorize
from ner_eval.methods.transformer import Transformer


class Tagger(Protocol):
    def fit(self, train: list[Sentence]) -> None: ...

    def predict(self, sents: list[list[str]]) -> list[list[str]]: ...


ENCODERS = {
    "bert": "google-bert/bert-base-cased",
    "roberta": "FacebookAI/roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "ettin": "jhu-clsp/ettin-encoder-150m",
}

METHODS: dict[str, Callable[..., Tagger]] = {
    "memorize": Memorize,
    "transformer": Transformer,
    **{name: partial(Transformer, model=m) for name, m in ENCODERS.items()},
}
