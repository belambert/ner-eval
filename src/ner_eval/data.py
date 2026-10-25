"""Dataset download, discovery, and loading."""

import io
import json
import os
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("NER_EVAL_DATA", "data"))
UNER_ORG = "UniversalNER"
OPENNER_RELEASE = "v0.9.0"
OPENNER_URL = "https://github.com/bltlab/open-ner/releases/download/{rel}/{var}.tar.gz"
OPENNER_VARIANTS = ("core_types", "standardized")
SPLITS = ("train", "dev", "test")


@dataclass
class Sentence:
    tokens: list[str]
    tags: list[str]


def list_datasets() -> list[str]:
    """Return ids of all downloaded datasets, e.g. `uner/en_ewt`."""
    ids = sorted({f"uner/{_uner_name(p)}" for p in (DATA_DIR / "uner").glob("*.iob2")})
    for var in OPENNER_VARIANTS:
        root = DATA_DIR / "openner" / var
        ids += [
            f"openner-{var}/{p.parent.name}/{p.name}" for p in sorted(root.glob("*/*"))
        ]
    return ids


def load(ds: str) -> dict[str, list[Sentence]]:
    """Load the available splits of a dataset id returned by `list_datasets`."""
    src, name = ds.split("/", 1)
    if src == "uner":
        paths = (DATA_DIR / "uner").glob(f"{name}*.iob2")
        return {
            _uner_split(p): read_uner(p) for p in sorted(paths) if _uner_name(p) == name
        }

    var = src.removeprefix("openner-")
    root = DATA_DIR / "openner" / var / name
    return {
        s: read_conll(root / f"{s}.txt") for s in SPLITS if (root / f"{s}.txt").exists()
    }


def read_uner(path: Path) -> list[Sentence]:
    """Read UNER's CoNLL-U-like IOB2 format (id, token, tag, ...), mapping OTH to O."""
    # OTH is annotated in some releases (hr_set, sr_set, sl_ssj) but isn't part of the benchmark
    return _read_blocks(
        path,
        lambda cols: (cols[1], "O" if cols[2].endswith("-OTH") else cols[2]),
        sep="\t",
    )


def read_conll(path: Path) -> list[Sentence]:
    """Read two-column `token tag` CoNLL files as used by OpenNER."""
    return _read_blocks(path, lambda cols: (cols[0], cols[1]), sep=" ")


def download_uner() -> None:
    """Download the IOB2 files from every UNER dataset repo."""
    out = DATA_DIR / "uner"
    out.mkdir(parents=True, exist_ok=True)
    repos = _gh_json(f"https://api.github.com/orgs/{UNER_ORG}/repos?per_page=100")
    for repo in sorted(r["name"] for r in repos if r["name"].startswith("UNER_")):
        for f in _gh_json(f"https://api.github.com/repos/{UNER_ORG}/{repo}/contents"):
            if f["name"].endswith(".iob2"):
                (out / f["name"]).write_bytes(_get(f["download_url"]))
                print(f"{repo}: {f['name']}")


def download_openner(var: str = "core_types") -> None:
    """Download and extract an OpenNER release variant."""
    assert var in OPENNER_VARIANTS, f"unknown variant: {var}"
    data = _get(OPENNER_URL.format(rel=OPENNER_RELEASE, var=var))
    out = DATA_DIR / "openner"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="data")
    print(f"openner-{var} -> {out / var}")


def _read_blocks(
    path: Path, parse: Callable[[list[str]], tuple[str, str]], sep: str
) -> list[Sentence]:
    sents: list[Sentence] = []
    toks: list[str] = []
    tags: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        if not line.strip():
            if toks:
                sents.append(Sentence(toks, tags))
                toks, tags = [], []
            continue
        tok, tag = parse(line.split(sep))
        toks.append(tok)
        tags.append(tag)

    if toks:
        sents.append(Sentence(toks, tags))
    return sents


# files are named like en_ewt-ud-dev.iob2; test-only sets may lack the suffix
def _uner_name(path: Path) -> str:
    return path.stem.split("-")[0]


def _uner_split(path: Path) -> str:
    return next((s for s in SPLITS if path.stem.endswith(f"-{s}")), "test")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url)
    if token := os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def _gh_json(url: str) -> list[dict]:
    return json.loads(_get(url))
