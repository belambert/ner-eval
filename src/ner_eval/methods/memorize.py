"""Lexicon baseline: tag token sequences seen as entities in training."""

from collections import Counter, defaultdict

from ner_eval.data import Sentence
from ner_eval.metrics import spans


class Memorize:
    """Greedy longest-match over training entity strings, using each string's most frequent type."""

    def __init__(self) -> None:
        self.lex: dict[tuple[str, ...], str] = {}
        self.max_len = 0

    def fit(self, train: list[Sentence]) -> None:
        counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for s in train:
            for i, j, t in spans(s.tags):
                counts[tuple(s.tokens[i:j])][t] += 1

        self.lex = {k: c.most_common(1)[0][0] for k, c in counts.items()}
        self.max_len = max(map(len, self.lex), default=0)

    def predict(self, sents: list[list[str]]) -> list[list[str]]:
        return [self._tag(toks) for toks in sents]

    def _tag(self, toks: list[str]) -> list[str]:
        tags = ["O"] * len(toks)
        i = 0
        while i < len(toks):
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                if t := self.lex.get(tuple(toks[i : i + n])):
                    tags[i : i + n] = [f"B-{t}"] + [f"I-{t}"] * (n - 1)
                    i += n - 1
                    break
            i += 1
        return tags
