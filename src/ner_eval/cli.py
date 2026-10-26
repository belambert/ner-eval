"""Command-line interface."""

import ast
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ner_eval import data, inventory
from ner_eval.methods import METHODS
from ner_eval.metrics import score

app = typer.Typer(no_args_is_help=True)


@app.command()
def evaluate(
    method: str,
    datasets: Annotated[
        list[str] | None, typer.Argument(help="coarse and/or fine; default both.")
    ] = None,
    split: str = "test",
    opt: Annotated[
        list[str] | None,
        typer.Option("-o", "--opt", help="Method option as key=value, e.g. epochs=1."),
    ] = None,
    max_eval: Annotated[
        int | None, typer.Option(help="Score only the first N sentences.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Append results to this JSONL file.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Train METHOD on each dataset's train split and score it on SPLIT."""
    assert method in METHODS, f"unknown method {method!r}; choose from {list(METHODS)}"
    opts = dict(_parse_opt(o) for o in opt or [])
    rows = []
    for ds in datasets or data.list_datasets():
        splits = data.load(ds)
        model = METHODS[method](**opts)
        model.fit(splits["train"])

        gold = splits[split][:max_eval]
        pred = model.predict([x.tokens for x in gold])
        s = score([x.tags for x in gold], pred)["ALL"]
        row = dict(method=method, opts=opts, dataset=ds, split=split, n=len(gold))
        rows.append(row | dict(p=s.p, r=s.r, f1=s.f1, tp=s.tp, fp=s.fp, fn=s.fn))
        if out:
            with out.open("a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
        if not as_json:
            typer.echo(f"{ds:<6} P={s.p:.3f} R={s.r:.3f} F1={s.f1:.3f}")

    if as_json:
        typer.echo(json.dumps(rows, indent=2))


@app.command("list")
def list_() -> None:
    """List downloaded datasets with sentence counts per split."""
    for ds in data.list_datasets():
        counts = " ".join(f"{s}={len(x)}" for s, x in data.load(ds).items())
        typer.echo(f"{ds:<6} {counts}")


@app.command()
def tags() -> None:
    """Print the entity-type inventory as a Markdown table (as embedded in the README)."""
    typer.echo(inventory.type_table())


@app.command()
def download() -> None:
    """Download Few-NERD into $NER_EVAL_DATA (default ./data/few-nerd)."""
    data.download()


def _parse_opt(o: str) -> tuple[str, Any]:
    """Parse key=value, reading the value as a Python literal when possible."""
    k, _, v = o.partition("=")
    try:
        return k, ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return k, v
