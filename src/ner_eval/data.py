"""Few-NERD and CleanCoNLL download and loading."""

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("NER_EVAL_DATA", "data"))
DATA_DIR = DATA_ROOT / "few-nerd"
CONLL_DIR = DATA_ROOT / "cleanconll"
# the supervised setting; inter/intra need episodic few-shot evaluation, which isn't supported
HF_URL = "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/supervised/{split}-00000-of-00001.parquet"
LEVELS = ("coarse", "fine")
# dataset id -> corpus name, the prefix of saved model and prediction names
CORPUS = {"coarse": "few-nerd", "fine": "few-nerd", "cleanconll": "conll"}
# the CleanCoNLL release builds the data by patching the original CoNLL-03 text
CONLL_REPO = "https://github.com/flairNLP/CleanCoNLL.git"
CONLL_COMMIT = "bedc569f0983ddd39c2c729fb5df64be710bcab4"
SPLITS = ("train", "dev", "test")
HF_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


@dataclass
class Sentence:
    tokens: list[str]
    tags: list[str]


def list_datasets() -> list[str]:
    """Return the ids of the downloaded datasets."""
    out = list(LEVELS) if (DATA_DIR / "test.parquet").exists() else []
    if (CONLL_DIR / "cleanconll.test").exists():
        out.append("cleanconll")
    return out


def run_name(method: str, ds: str) -> str:
    """Name a method's model or predictions on a dataset, e.g. `few-nerd-bert-coarse`."""
    return f"{CORPUS[ds]}-{method}-{ds}"


def load(ds: str, splits: tuple[str, ...] = SPLITS) -> dict[str, list[Sentence]]:
    """Load splits of Few-NERD's `coarse` or `fine` labels (IO converted to BIO) or `cleanconll`."""
    assert ds in CORPUS, f"unknown dataset {ds!r}; choose from {list(CORPUS)}"
    assert set(splits) <= set(SPLITS), f"unknown split; choose from {SPLITS}"
    if ds == "cleanconll":
        return {s: read_conll(CONLL_DIR / f"cleanconll.{s}") for s in splits}
    return {s: read_parquet(DATA_DIR / f"{s}.parquet", ds) for s in splits}


def read_parquet(path: Path, lvl: str = "fine") -> list[Sentence]:
    """Read a Hugging Face Few-NERD parquet file at the given label granularity."""
    table = pq.read_table(path)
    col = "ner_tags" if lvl == "coarse" else "fine_ner_tags"
    names = label_names(table, col)
    return [
        Sentence(r["tokens"], io_to_bio([names[i] for i in r[col]]))
        for r in table.select(["tokens", col]).to_pylist()
    ]


def label_names(table: pa.Table, col: str) -> list[str]:
    """Return the ClassLabel names that Hugging Face stores in the parquet metadata."""
    info = json.loads(table.schema.metadata[b"huggingface"])["info"]
    return info["features"][col]["feature"]["names"]


def read_conll(path: Path) -> list[Sentence]:
    """Read a CleanCoNLL file: tab-separated columns with final BIO tags last."""
    out: list[Sentence] = []
    for block in path.read_text().split("\n\n"):
        rows = [line.split("\t") for line in block.splitlines()]
        if rows and rows[0][0] != "-DOCSTART-":
            out.append(Sentence([r[0] for r in rows], [r[-1] for r in rows]))
    return out


def io_to_bio(tags: list[str]) -> list[str]:
    """Convert IO tags to BIO; adjacent same-type entities merge, as IO can't separate them."""
    return [
        "O" if t == "O" else ("I-" if i and tags[i - 1] == t else "B-") + t
        for i, t in enumerate(tags)
    ]


def download(corpus: str) -> None:
    """Download `few-nerd` or build `cleanconll` under the data directory."""
    assert corpus in ("few-nerd", "cleanconll"), f"unknown corpus {corpus!r}"
    if corpus == "cleanconll":
        return build_cleanconll()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, hf in HF_SPLITS.items():
        urllib.request.urlretrieve(
            HF_URL.format(split=hf), DATA_DIR / f"{split}.parquet"
        )


def build_cleanconll() -> None:
    """Build CleanCoNLL with its release script, which fetches and patches CoNLL-03."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "CleanCoNLL"
        subprocess.run(["git", "clone", "-q", CONLL_REPO, repo], check=True)
        subprocess.run(["git", "-C", repo, "checkout", "-q", CONLL_COMMIT], check=True)
        subprocess.run(
            ["bash", "create_cleanconll_from_conll03.sh"], cwd=repo, check=True
        )
        CONLL_DIR.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            shutil.copy(repo / f"data/cleanconll/cleanconll.{s}", CONLL_DIR)
