"""Entity-type inventory rendered as a Markdown table."""

from collections import Counter

from ner_eval import data
from ner_eval.metrics import spans


def type_table() -> str:
    """One row per coarse type with its fine types and mention counts over all splits."""
    sents = [s for ss in data.load("fine").values() for s in ss]
    fine = Counter(t for s in sents for *_, t in spans(s.tags))
    coarse: Counter[str] = Counter()
    for t, n in fine.items():
        coarse[t.split("-")[0]] += n

    rows = []
    for c in sorted(coarse):
        subs = [
            f"{t.split('-', 1)[1]} ({fine[t]:,})"
            for t in sorted(fine)
            if t.startswith(f"{c}-")
        ]
        rows.append([c, f"{coarse[c]:,}", ", ".join(subs)])
    return md_table(["Coarse type", "Mentions", "Fine types (mentions)"], rows)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a visually aligned Markdown table."""
    w = [max(map(len, col)) for col in zip(header, *rows)]
    fmt = lambda r: "| " + " | ".join(c.ljust(n) for c, n in zip(r, w)) + " |"
    return "\n".join([fmt(header), fmt(["-" * n for n in w]), *map(fmt, rows)])
