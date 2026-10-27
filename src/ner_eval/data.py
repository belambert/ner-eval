"""CleanCoNLL download and loading."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(os.environ.get("NER_EVAL_DATA", "data"))
CONLL_DIR = DATA_ROOT / "cleanconll"
# dataset id -> corpus name, the prefix of saved model and prediction names
CORPUS = {"cleanconll": "conll"}
# the CleanCoNLL release builds the data by patching the original CoNLL-03 text
CONLL_REPO = "https://github.com/flairNLP/CleanCoNLL.git"
CONLL_COMMIT = "bedc569f0983ddd39c2c729fb5df64be710bcab4"
SPLITS = ("train", "dev", "test")


@dataclass
class Sentence:
    tokens: list[str]
    tags: list[str]


def list_datasets() -> list[str]:
    """Return the ids of the downloaded datasets."""
    return ["cleanconll"] if (CONLL_DIR / "cleanconll.test").exists() else []


def run_name(method: str, ds: str) -> str:
    """Name a method's model or predictions on a dataset, e.g. `conll-bert-cleanconll`."""
    return f"{CORPUS[ds]}-{method}-{ds}"


def load(ds: str, splits: tuple[str, ...] = SPLITS) -> dict[str, list[Sentence]]:
    """Load splits of a dataset."""
    assert ds in CORPUS, f"unknown dataset {ds!r}; choose from {list(CORPUS)}"
    assert set(splits) <= set(SPLITS), f"unknown split; choose from {SPLITS}"
    return {s: read_conll(CONLL_DIR / f"cleanconll.{s}") for s in splits}


def read_conll(path: Path) -> list[Sentence]:
    """Read a CleanCoNLL file: tab-separated columns with final BIO tags last."""
    out: list[Sentence] = []
    for block in path.read_text().split("\n\n"):
        rows = [line.split("\t") for line in block.splitlines()]
        if rows and rows[0][0] != "-DOCSTART-":
            out.append(Sentence([r[0] for r in rows], [r[-1] for r in rows]))
    return out


def download() -> None:
    """Build CleanCoNLL with its release script, which fetches and patches CoNLL-03."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "CleanCoNLL"
        subprocess.run(["git", "clone", "-q", CONLL_REPO, repo], check=True)
        subprocess.run(["git", "-C", repo, "checkout", "-q", CONLL_COMMIT], check=True)
        subprocess.run(
            ["bash", "create_cleanconll_from_conll03.sh"], cwd=repo, check=True
        )
        CONLL_DIR.mkdir(parents=True, exist_ok=True)
        for s, orig in zip(SPLITS, ("train", "valid", "test")):
            shutil.copy(repo / f"data/cleanconll/cleanconll.{s}", CONLL_DIR)
            # keep the original labels too, for `conll-diff`
            shutil.copy(repo / f"data/conll03/{orig}.txt", CONLL_DIR / f"conll03.{s}")
