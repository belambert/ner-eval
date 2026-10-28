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
    "roberta-large": "FacebookAI/roberta-large",
    "modernbert": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "deberta": "microsoft/deberta-v3-base",
    "deberta-large": "microsoft/deberta-v3-large",
    "ettin": "jhu-clsp/ettin-encoder-150m",
    **{f"ettin-{s}": f"jhu-clsp/ettin-encoder-{s}" for s in ["400m", "1b"]},
}

# decoders, with the learning rate where it differs from the default
DECODERS = {
    **{
        f"ettin-dec-{s}": (f"jhu-clsp/ettin-decoder-{s}", 5e-5)
        for s in ["150m", "400m", "1b"]
    },
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", 2e-5),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", 2e-5),
}

METHODS: dict[str, Callable[..., Tagger]] = {
    "memorize": Memorize,
    "transformer": Transformer,
    **{name: partial(Transformer, model=m) for name, m in ENCODERS.items()},
    **{
        name: partial(Transformer, model=m, lr=lr, subword="last")
        for name, (m, lr) in DECODERS.items()
    },
}


def load(method: str, path: str) -> Tagger:
    """Load a saved tagger for METHOD; transformers load from a local dir or Hugging Face repo."""
    # transformer methods are partials of Transformer, which records its own options when saved
    f = METHODS[method]
    return getattr(f, "func", f).load(path)
