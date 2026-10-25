"""Exact-match span-level precision, recall, and F1."""

from collections import Counter, defaultdict
from dataclasses import dataclass

Span = tuple[int, int, str]


@dataclass
class Scores:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def p(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def r(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.p * self.r / (self.p + self.r) if self.p + self.r else 0.0


def score(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Scores]:
    """Score tag sequences; returns per-type scores plus a micro `ALL` entry."""
    assert len(gold) == len(pred), "gold/pred sentence count mismatch"
    res: defaultdict[str, Scores] = defaultdict(Scores)
    for g, p in zip(gold, pred, strict=True):
        assert len(g) == len(p), "gold/pred token count mismatch"
        gs, ps = Counter(spans(g)), Counter(spans(p))
        for (*_, t), n in (gs & ps).items():
            res[t].tp += n
        for (*_, t), n in (ps - gs).items():
            res[t].fp += n
        for (*_, t), n in (gs - ps).items():
            res[t].fn += n

    total = Scores()
    for s in res.values():
        total.tp, total.fp, total.fn = total.tp + s.tp, total.fp + s.fp, total.fn + s.fn
    return {"ALL": total, **dict(sorted(res.items()))}


def spans(tags: list[str]) -> list[Span]:
    """Extract (start, end, type) spans from BIO tags; stray I- tags start a span (conlleval)."""
    out: list[Span] = []
    start, typ = -1, ""
    for i, tag in enumerate([*tags, "O"]):
        pre, _, t = tag.partition("-")
        if typ and (pre != "I" or t != typ):
            out.append((start, i, typ))
            typ = ""
        if pre == "B" or (pre == "I" and not typ):
            start, typ = i, t
    return out
