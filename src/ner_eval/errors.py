"""Render a model's saved predictions as an HTML page of its errors."""

import json
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

import typer

from ner_eval import data
from ner_eval.conll_diff import changes
from ner_eval.metrics import score, spans


def main() -> None:
    """Run the `ner-errors` command."""
    typer.run(_cli)


def _cli(
    preds: Annotated[
        Path, typer.Argument(help="Prediction file written with `--preds`.")
    ],
    out: Annotated[Path, typer.Argument(help="HTML file to write.")] = Path(
        "site/errors.html"
    ),
) -> None:
    """Write an HTML page of every sentence where the predictions differ from gold."""
    write_page(preds, out)


def write_page(preds: Path, out: Path) -> None:
    """Write the error page for the predictions in `preds` to `out`."""
    page = files("ner_eval").joinpath("errors.html").read_text()
    # escape "</" so the data can't close its script tag
    js = json.dumps(errors(preds), separators=(",", ":")).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("__DATA__", js))


def errors(preds: Path) -> dict[str, Any]:
    """Return scores, error counts, type confusions, and the sentences with errors."""
    # the file is named CORPUS-METHOD-DATASET-SPLIT, and METHOD may contain dashes
    name, ds, split = preds.stem.rsplit("-", 2)
    method = name.removeprefix(f"{data.CORPUS[ds]}-")
    pred = [json.loads(line) for line in preds.open()]
    gold = data.load(ds, (split,))[split][: len(pred)]

    sents, kinds, trans = [], Counter[str](), Counter[tuple[str, str]]()
    for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
        gs, ps = spans(g.tags), spans(p)
        if gs == ps:
            continue
        cs = changes(gs, ps)
        sents.append(dict(i=i, tokens=g.tokens, gold=gs, pred=ps, changes=cs))
        kinds.update(c[0] for c in cs)
        trans.update((c[1][2], c[2][2]) for c in cs if c[0] == "type")

    scores = score([x.tags for x in gold], pred)
    return dict(
        method=method,
        dataset=ds,
        split=split,
        n=len(gold),
        scores={t: [s.p, s.r, s.f1, s.tp + s.fn] for t, s in scores.items()},
        kinds=kinds,
        trans=[[a, b, n] for (a, b), n in trans.most_common()],
        sents=sents,
    )
