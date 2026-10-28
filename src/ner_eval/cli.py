"""Command-line interface."""

import ast
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ner_eval import data, methods
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
Datasets = Annotated[
    list[str] | None,
    typer.Argument(help="Dataset ids, e.g. cleanconll; default all downloaded."),
]
Models = Annotated[
    str,
    typer.Option(
        help="Model directory, holding one MODELS/CORPUS-METHOD-DATASET per run; "
        "for evaluate, also a Hugging Face user or org."
    ),
]
MaxEval = Annotated[int | None, typer.Option(help="Score only the first N sentences.")]
Out = Annotated[Path | None, typer.Option(help="Append results to this JSONL file.")]
Preds = Annotated[
    Path | None,
    typer.Option(
        help="Write predicted tags to PREDS/CORPUS-METHOD-DATASET-SPLIT.jsonl "
        "for `rescore`."
    ),
]
Json = Annotated[bool, typer.Option("--json")]


@app.command()
def train(
    method: str,
    datasets: Datasets = None,
    opt: Annotated[
        list[str] | None,
        typer.Option("-o", "--opt", help="Method option as key=value, e.g. epochs=1."),
    ] = None,
    models: Models = "results/models",
    split: str = "test",
    max_eval: MaxEval = None,
    out: Out = None,
    preds: Preds = None,
    overlap: Overlap = 1.0,
    as_json: Json = False,
) -> None:
    """Train METHOD on each dataset's train split, save it to MODELS, and score it on SPLIT."""
    assert method in METHODS, f"unknown method {method!r}; choose from {list(METHODS)}"
    opts = dict(_parse_opt(o) for o in opt or [])
    rows = []
    for ds in datasets or data.list_datasets():
        model = METHODS[method](**opts)
        model.fit(data.load(ds, ("train",))["train"])
        ckpt = f"{models}/{data.run_name(method, ds)}"
        model.save(Path(ckpt))
        row = _evaluate(model, method, ds, split, max_eval, preds, overlap, as_json)
        rows.append(dict(opts=opts, ckpt=ckpt) | row)
        _write(rows[-1], out)

    if as_json:
        typer.echo(json.dumps(rows, indent=2))


@app.command()
def evaluate(
    method: str,
    datasets: Datasets = None,
    models: Models = "results/models",
    split: str = "test",
    max_eval: MaxEval = None,
    out: Out = None,
    preds: Preds = None,
    overlap: Overlap = 1.0,
    as_json: Json = False,
) -> None:
    """Score METHOD's trained models from MODELS on each dataset's SPLIT."""
    assert method in METHODS, f"unknown method {method!r}; choose from {list(METHODS)}"
    rows = []
    for ds in datasets or data.list_datasets():
        ckpt = f"{models}/{data.run_name(method, ds)}"
        model = methods.load(method, ckpt)
        row = _evaluate(model, method, ds, split, max_eval, preds, overlap, as_json)
        rows.append(dict(ckpt=ckpt) | row)
        _write(rows[-1], out)

    if as_json:
        typer.echo(json.dumps(rows, indent=2))


@app.command()
def rescore(
    files: Annotated[
        list[Path],
        typer.Argument(
            help="Prediction files written by `train` or `evaluate` with `--preds`."
        ),
    ],
    overlap: Overlap = 1.0,
    as_json: Json = False,
) -> None:
    """Score saved predictions against gold tags without rerunning the model."""
    rows = []
    for path in files:
        # the dataset and split are the last two parts of CORPUS-METHOD-DATASET-SPLIT
        *_, ds, split = path.stem.rsplit("-", 2)
        pred = [json.loads(line) for line in path.open()]
        # predictions cover the first N sentences when run with --max-eval
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
def download() -> None:
    """Download the data into $NER_EVAL_DATA (default ./data)."""
    data.download()


def _evaluate(
    model: methods.Tagger,
    method: str,
    ds: str,
    split: str,
    max_eval: int | None,
    preds: Path | None,
    overlap: float,
    quiet: bool,
) -> dict[str, Any]:
    """Predict on SPLIT, optionally save the predictions, and return a result row."""
    gold = data.load(ds, (split,))[split][:max_eval]
    pred = model.predict([x.tokens for x in gold])
    if preds:
        preds.mkdir(parents=True, exist_ok=True)
        path = preds / f"{data.run_name(method, ds)}-{split}.jsonl"
        with path.open("w") as f:
            f.writelines(json.dumps(p, separators=(",", ":")) + "\n" for p in pred)

    row = dict(method=method, dataset=ds, split=split, n=len(gold), overlap=overlap)
    return row | _score(f"{ds:<10}", gold, pred, overlap, quiet)


def _write(row: dict[str, Any], out: Path | None) -> None:
    """Append ROW to the JSONL file OUT, if given."""
    if out:
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


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
