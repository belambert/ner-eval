"""Entity-type inventories rendered as Markdown tables."""

from collections import Counter

from ner_eval import data
from ner_eval.metrics import spans


def uner_table() -> str:
    """One row per UNER treebank with splits and entity counts per type."""
    types = ["PER", "LOC", "ORG"]
    rows = []
    for ds in (d for d in data.list_datasets() if d.startswith("uner/")):
        splits = data.load(ds)
        c = _count(splits)
        assert set(c) <= set(types), f"unexpected types in {ds}: {set(c) - set(types)}"
        order = "/".join(s for s in data.SPLITS if s in splits)
        rows.append([f"`{ds}`", order, *(str(c[t]) for t in types)])
    return md_table(["Dataset", "Splits", *types], rows)


def openner_table(var: str = "standardized") -> str:
    """One row per OpenNER corpus and type set; languages that share a type set are merged."""
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for ds in (d for d in data.list_datasets() if d.startswith(f"openner-{var}/")):
        _, corpus, lang = ds.split("/")
        types = tuple(sorted(_count(data.load(ds))))
        groups.setdefault((corpus, types), []).append(lang)

    rows = [
        [corpus, ", ".join(langs), str(len(types)), ", ".join(types)]
        for (corpus, types), langs in sorted(
            groups.items(), key=lambda x: x[0][0].lower()
        )
    ]
    return md_table(["Corpus", "Languages", "#", "Types"], rows)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a visually aligned Markdown table."""
    w = [max(map(len, col)) for col in zip(header, *rows)]
    fmt = lambda r: "| " + " | ".join(c.ljust(n) for c, n in zip(r, w)) + " |"
    return "\n".join([fmt(header), fmt(["-" * n for n in w]), *map(fmt, rows)])


def _count(splits: dict[str, list[data.Sentence]]) -> Counter[str]:
    return Counter(t for ss in splits.values() for s in ss for *_, t in spans(s.tags))
