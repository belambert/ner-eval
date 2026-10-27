"""Few-NERD download and loading."""

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("NER_EVAL_DATA", "data/few-nerd"))
# the supervised setting; inter/intra need episodic few-shot evaluation, which isn't supported
HF_URL = "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/supervised/{split}-00000-of-00001.parquet"
LEVELS = ("coarse", "fine")
SPLITS = ("train", "dev", "test")
HF_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


@dataclass
class Sentence:
    tokens: list[str]
    tags: list[str]


def list_datasets() -> list[str]:
    """Return the dataset ids (label granularities) if the data is downloaded."""
    return list(LEVELS) if (DATA_DIR / "test.parquet").exists() else []


def load(ds: str, splits: tuple[str, ...] = SPLITS) -> dict[str, list[Sentence]]:
    """Load splits of `coarse` or `fine`, with Few-NERD's IO tags converted to BIO."""
    assert ds in LEVELS, f"unknown dataset {ds!r}; choose from {LEVELS}"
    assert set(splits) <= set(SPLITS), f"unknown split; choose from {SPLITS}"
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


def io_to_bio(tags: list[str]) -> list[str]:
    """Convert IO tags to BIO; adjacent same-type entities merge, as IO can't separate them."""
    return [
        "O" if t == "O" else ("I-" if i and tags[i - 1] == t else "B-") + t
        for i, t in enumerate(tags)
    ]


def download() -> None:
    """Download the parquet file for each split."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, hf in HF_SPLITS.items():
        urllib.request.urlretrieve(
            HF_URL.format(split=hf), DATA_DIR / f"{split}.parquet"
        )
