"""NER methods and their registry."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Protocol

from ner_eval.data import Sentence
from ner_eval.methods.memorize import Memorize
from ner_eval.methods.transformer import Transformer


class Tagger(Protocol):
    def fit(self, train: list[Sentence]) -> None: ...

    def predict(self, sents: list[list[str]]) -> list[list[str]]: ...

    def save(self, path: Path) -> None: ...


ENCODERS = {
    "bert": "google-bert/bert-base-cased",
    "roberta": "FacebookAI/roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "ettin": "jhu-clsp/ettin-encoder-150m",
    **{
        f"ettin-{s}": f"jhu-clsp/ettin-encoder-{s}"
        for s in ["17m", "32m", "68m", "400m", "1b"]
    },
}

METHODS: dict[str, Callable[..., Tagger]] = {
    "memorize": Memorize,
    "transformer": Transformer,
    **{name: partial(Transformer, model=m) for name, m in ENCODERS.items()},
}


def load(method: str, path: str) -> Tagger:
    """Load a saved tagger for METHOD; encoders load from a local dir or Hugging Face repo."""
    # encoder methods are partials of Transformer, which records its own options when saved
    f = METHODS[method]
    return getattr(f, "func", f).load(path)
