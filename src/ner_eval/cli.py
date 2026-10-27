"""Command-line interface."""

import ast
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ner_eval import data, inventory, methods
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
    save: Annotated[
        Path | None,
        typer.Option(help="Save each trained model to SAVE/few-nerd-METHOD-DATASET."),
    ] = None,
    load: Annotated[
        str | None,
        typer.Option(
            help="Load models from LOAD/few-nerd-METHOD-DATASET instead of training; "
            "LOAD is a local directory or a Hugging Face user or org."
        ),
    ] = None,
    overlap: Annotated[
        float,
        typer.Option(
            help="Count a predicted span as correct if its token IoU with a gold span "
            "of the same type is at least this; 1.0 is exact match."
        ),
    ] = 1.0,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Train METHOD on each dataset's train split (or load it) and score it on SPLIT."""
    assert method in METHODS, f"unknown method {method!r}; choose from {list(METHODS)}"
    opts = dict(_parse_opt(o) for o in opt or [])
    assert not (load and opts), "options can't be applied to a loaded model"
    rows = []
    for ds in datasets or data.list_datasets():
        splits = data.load(ds)
        name = f"few-nerd-{method}-{ds}"
        if load:
            model = methods.load(method, f"{load}/{name}")
        else:
            model = METHODS[method](**opts)
            model.fit(splits["train"])
        if save:
            model.save(save / name)

        gold = splits[split][:max_eval]
        pred = model.predict([x.tokens for x in gold])
        s = score([x.tags for x in gold], pred, overlap)["ALL"]
        row = dict(
            method=method,
            opts=opts,
            dataset=ds,
            split=split,
            n=len(gold),
            overlap=overlap,
        )
        if load:
            row["ckpt"] = f"{load}/{name}"
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
