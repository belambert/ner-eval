"""Command-line interface."""

import json
from typing import Annotated

import typer

from ner_eval import data, inventory
from ner_eval.methods import METHODS
from ner_eval.metrics import score

app = typer.Typer(no_args_is_help=True)


@app.command()
def evaluate(
    method: str,
    datasets: Annotated[
        list[str] | None, typer.Argument(help="Dataset ids; default all.")
    ] = None,
    train_on: Annotated[
        str | None, typer.Option(help="Train on this dataset instead.")
    ] = None,
    split: str = "test",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Train METHOD on each dataset's train split and score it on SPLIT."""
    assert method in METHODS, f"unknown method {method!r}; choose from {list(METHODS)}"
    rows = []
    for ds in datasets or data.list_datasets():
        splits = data.load(ds)
        train = data.load(train_on)["train"] if train_on else splits.get("train")
        if train is None or split not in splits:
            typer.echo(f"skip {ds}: missing train or {split} split", err=True)
            continue

        model = METHODS[method]()
        model.fit(train)
        gold = splits[split]
        s = score([x.tags for x in gold], model.predict([x.tokens for x in gold]))[
            "ALL"
        ]
        rows.append(dict(dataset=ds, p=s.p, r=s.r, f1=s.f1, tp=s.tp, fp=s.fp, fn=s.fn))
        if not as_json:
            typer.echo(f"{ds:<50} P={s.p:.3f} R={s.r:.3f} F1={s.f1:.3f}")

    if as_json:
        typer.echo(json.dumps(rows, indent=2))


@app.command("list")
def list_() -> None:
    """List downloaded datasets with sentence counts per split."""
    for ds in data.list_datasets():
        counts = " ".join(f"{s}={len(x)}" for s, x in data.load(ds).items())
        typer.echo(f"{ds:<50} {counts}")


@app.command()
def tags() -> None:
    """Print entity-type inventories as Markdown tables (as embedded in the README)."""
    typer.echo(inventory.uner_table())
    typer.echo()
    typer.echo(inventory.openner_table())


@app.command()
def download(
    source: Annotated[str, typer.Argument(help="uner, openner, or all")] = "all",
    variant: Annotated[str, typer.Option(help="OpenNER variant.")] = "core_types",
) -> None:
    """Download datasets into $NER_EVAL_DATA (default ./data)."""
    if source in ("uner", "all"):
        data.download_uner()
    if source in ("openner", "all"):
        data.download_openner(variant)
