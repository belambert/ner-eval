"""Span-level precision, recall, and F1, with exact or overlap matching."""

from collections import defaultdict
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


def score(
    gold: list[list[str]], pred: list[list[str]], overlap: float = 1.0
) -> dict[str, Scores]:
    """Score tag sequences; returns per-type scores plus a micro `ALL` entry.

    A predicted span matches a gold span of the same type when their token IoU
    (intersection over union) is at least `overlap`; 1.0 is exact match.
    """
    assert len(gold) == len(pred), "gold/pred sentence count mismatch"
    assert 0 < overlap <= 1, "overlap must be in (0, 1]"
    res: defaultdict[str, Scores] = defaultdict(Scores)
    for g, p in zip(gold, pred, strict=True):
        assert len(g) == len(p), "gold/pred token count mismatch"
        gs, ps = spans(g), spans(p)
        mg, mp = _match(gs, ps, overlap)
        for i, (*_, t) in enumerate(gs):
            if i in mg:
                res[t].tp += 1
            else:
                res[t].fn += 1
        for j, (*_, t) in enumerate(ps):
            if j not in mp:
                res[t].fp += 1

    total = Scores()
    for s in res.values():
        total.tp, total.fp, total.fn = total.tp + s.tp, total.fp + s.fp, total.fn + s.fn
    return {"ALL": total, **dict(sorted(res.items()))}


def iou(a: Span, b: Span) -> float:
    """Token intersection over union of two spans, ignoring type."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (a[1] - a[0] + b[1] - b[0] - inter)


def _match(gs: list[Span], ps: list[Span], overlap: float) -> tuple[set[int], set[int]]:
    """Pair gold and predicted spans one-to-one, best overlap first; return matched indices."""
    pairs = sorted(
        (
            (o, i, j)
            for i, g in enumerate(gs)
            for j, p in enumerate(ps)
            if g[2] == p[2] and (o := iou(g, p)) >= overlap
        ),
        reverse=True,
    )
    mg: set[int] = set()
    mp: set[int] = set()
    for _, i, j in pairs:
        if i not in mg and j not in mp:
            mg.add(i)
            mp.add(j)
    return mg, mp


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
