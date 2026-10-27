"""Compare CoNLL-03 and CleanCoNLL labels and render the differences as an HTML page."""

import difflib
import json
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

from ner_eval.data import CONLL_DIR, SPLITS
from ner_eval.metrics import Span, spans

Doc = list[list[tuple[str, str]]]


def write_page(out: Path) -> None:
    """Write the diff page for all splits to `out`."""
    page = files("ner_eval").joinpath("conll_diff.html").read_text()
    # escape "</" so the data can't close its script tag
    data = json.dumps(diff(), separators=(",", ":")).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("__DATA__", data))


def diff() -> dict[str, Any]:
    """Return per-split counts, changed sentences with their changes, and type transitions."""
    res: dict[str, Any] = {"splits": {}, "sents": [], "kinds": Counter()}
    trans: Counter[tuple[str, str]] = Counter()
    for split in SPLITS:
        orig = read_docs(CONLL_DIR / f"conll03.{split}")
        clean = read_docs(CONLL_DIR / f"cleanconll.{split}")
        assert len(orig) == len(clean), f"{split}: document counts differ"
        stats: Counter[str] = Counter()
        for i, (od, cd) in enumerate(zip(orig, clean)):
            for sent in diff_doc(od, cd, stats):
                res["sents"].append({"split": split, "doc": i} | sent)
                for kind, o, c in sent["changes"]:
                    res["kinds"][f"{kind}|{split}"] += 1
                    if kind == "type":
                        trans[o[2], c[2]] += 1
        res["splits"][split] = dict(stats)
    res["trans"] = [[a, b, n] for (a, b), n in trans.most_common()]
    return res


def read_docs(path: Path) -> list[Doc]:
    """Read a CoNLL-style file into documents of sentences of (token, last-column tag)."""
    docs: list[Doc] = []
    for block in path.read_text().split("\n\n"):
        rows = [line.split() for line in block.splitlines() if line.strip()]
        if rows and rows[0][0] == "-DOCSTART-":
            docs.append([])
            rows = rows[1:]
        if rows:
            docs[-1].append([(r[0], r[-1]) for r in rows])
    return docs


def diff_doc(od: Doc, cd: Doc, stats: Counter[str]) -> list[dict[str, Any]]:
    """Return the CleanCoNLL sentences of a document whose entities differ from CoNLL-03."""
    otoks = [t for s in od for t, _ in s]
    ctoks = [t for s in cd for t, _ in s]
    # CleanCoNLL fixes some tokens and sentence splits, so map positions through an alignment
    pos = {}
    sm = difflib.SequenceMatcher(None, otoks, ctoks, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        pos |= {a + k: b + k for k in range(n)}
    oents = set()
    for s, e, t in spans([g for sent in od for _, g in sent]):
        if s in pos and e - 1 in pos:
            oents.add((pos[s], pos[e - 1] + 1, t))
        else:
            stats["unaligned"] += 1

    out, off = [], 0
    for sent in cd:
        n = len(sent)
        cents = {(s + off, e + off, t) for s, e, t in spans([g for _, g in sent])}
        ents = {x for x in oents if off <= x[0] < off + n}
        stats["sentences"] += 1
        stats["entities_orig"] += len(ents)
        stats["entities_clean"] += len(cents)
        if ents != cents:
            rel = lambda xs: sorted((s - off, e - off, t) for s, e, t in xs)
            o, c = rel(ents), rel(cents)
            out.append(
                {"tokens": [t for t, _ in sent], "orig": o, "clean": c}
                | {"changes": changes(o, c)}
            )
        off += n
    return out


def changes(orig: list[Span], clean: list[Span]) -> list[list[Any]]:
    """Pair differing entities as [kind, orig, clean]; kind is type, boundary, added, or removed.

    An entity overlapping any changed entity on the other side is a boundary change, so a
    merge or split counts each of its parts.
    """
    ov = lambda a, b: a[0] < b[1] and b[0] < a[1]
    os = [x for x in orig if x not in clean]
    cs = [x for x in clean if x not in orig]
    out: list[list[Any]] = []
    for o in os:
        same = [c for c in cs if c[:2] == o[:2]]
        hits = same or [c for c in cs if ov(o, c)]
        kind = "type" if same else "boundary" if hits else "removed"
        out.append([kind, o, hits[0] if hits else None])
    paired = {x[2] for x in out}
    for c in cs:
        if c not in paired:
            hits = [o for o in os if ov(o, c)]
            out.append(["boundary", hits[0], c] if hits else ["added", None, c])
    return out
