"""Command-line interface."""

import ast
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ner_eval import conll_diff, data, inventory, methods
from ner_eval.methods import METHODS
from ner_eval.metrics import score

app = typer.Typer(no_args_is_help=True)

Overlap = Annotated[
    float,
    typer.Option(
        help="Count a predicted span as correct if its token IoU with a gold span "
        "of the same type is at least this; 1.0 is exact match."
    ),
]


@app.command()
def evaluate(
    method: str,
    datasets: Annotated[
        list[str] | None,
        typer.Argument(help="coarse, fine, and/or cleanconll; default all downloaded."),
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
        typer.Option(help="Save each trained model to SAVE/CORPUS-METHOD-DATASET."),
    ] = None,
    load: Annotated[
        str | None,
        typer.Option(
            help="Load models from LOAD/CORPUS-METHOD-DATASET instead of training; "
            "LOAD is a local directory or a Hugging Face user or org."
        ),
    ] = None,
    preds: Annotated[
        Path | None,
        typer.Option(
            help="Write predicted tags to PREDS/CORPUS-METHOD-DATASET-SPLIT.jsonl "
            "for `rescore`."
        ),
    ] = None,
    overlap: Overlap = 1.0,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Train METHOD on each dataset's train split (or load it) and score it on SPLIT."""
    assert method in METHODS, f"unknown method {method!r}; choose from {list(METHODS)}"
    opts = dict(_parse_opt(o) for o in opt or [])
    assert not (load and opts), "options can't be applied to a loaded model"
    rows = []
    for ds in datasets or data.list_datasets():
        splits = data.load(ds)
        name = data.run_name(method, ds)
        if load:
            model = methods.load(method, f"{load}/{name}")
        else:
            model = METHODS[method](**opts)
            model.fit(splits["train"])
        if save:
            model.save(save / name)

        gold = splits[split][:max_eval]
        pred = model.predict([x.tokens for x in gold])
        if preds:
            preds.mkdir(parents=True, exist_ok=True)
            with (preds / f"{name}-{split}.jsonl").open("w") as f:
                f.writelines(json.dumps(p, separators=(",", ":")) + "\n" for p in pred)

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
        rows.append(row | _score(f"{ds:<10}", gold, pred, overlap, as_json))
        if out:
            with out.open("a") as f:
                f.write(json.dumps(rows[-1]) + "\n")

    if as_json:
        typer.echo(json.dumps(rows, indent=2))


@app.command()
def rescore(
    files: Annotated[
        list[Path],
        typer.Argument(help="Prediction files written by `evaluate --preds`."),
    ],
    overlap: Overlap = 1.0,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Score saved predictions against gold tags without rerunning the model."""
    rows = []
    for path in files:
        # the dataset and split are the last two parts of CORPUS-METHOD-DATASET-SPLIT
        *_, ds, split = path.stem.rsplit("-", 2)
        pred = [json.loads(line) for line in path.open()]
        # predictions cover the first N sentences when evaluate ran with --max-eval
        gold = data.load(ds, (split,))[split][: len(pred)]
        assert len(gold) == len(pred), f"{path} has more sentences than {ds}/{split}"
        row = dict(
            file=str(path), dataset=ds, split=split, n=len(pred), overlap=overlap
        )
        rows.append(row | _score(path.stem, gold, pred, overlap, as_json))

    if as_json:
        typer.echo(json.dumps(rows, indent=2))


@app.command("list")
def list_() -> None:
    """List downloaded datasets with sentence counts per split."""
    for ds in data.list_datasets():
        counts = " ".join(f"{s}={len(x)}" for s, x in data.load(ds).items())
        typer.echo(f"{ds:<10} {counts}")


@app.command()
def tags() -> None:
    """Print the entity-type inventory as a Markdown table (as embedded in docs/few-nerd.md)."""
    typer.echo(inventory.type_table())


@app.command("conll-diff")
def conll_diff_(
    out: Annotated[Path, typer.Argument(help="HTML file to write.")] = Path(
        "site/index.html"
    ),
) -> None:
    """Write an HTML page of the entity labels CleanCoNLL changed from CoNLL-03."""
    conll_diff.write_page(out)


@app.command()
def download(
    corpora: Annotated[
        list[str] | None,
        typer.Argument(help="few-nerd and/or cleanconll; default both."),
    ] = None,
) -> None:
    """Download the data into $NER_EVAL_DATA (default ./data)."""
    for c in corpora or ["few-nerd", "cleanconll"]:
        data.download(c)


def _score(
    label: str,
    gold: list[data.Sentence],
    pred: list[list[str]],
    overlap: float,
    quiet: bool,
) -> dict[str, Any]:
    """Score predictions, print a summary line unless `quiet`, and return the micro scores."""
    s = score([x.tags for x in gold], pred, overlap)["ALL"]
    if not quiet:
        typer.echo(f"{label} P={s.p:.3f} R={s.r:.3f} F1={s.f1:.3f}")
    return dict(p=s.p, r=s.r, f1=s.f1, tp=s.tp, fp=s.fp, fn=s.fn)


def _parse_opt(o: str) -> tuple[str, Any]:
    """Parse key=value, reading the value as a Python literal when possible."""
    k, _, v = o.partition("=")
    try:
        return k, ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return k, v
