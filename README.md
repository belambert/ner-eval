# ner-eval

A harness for comparing named entity recognition methods on recent multilingual
benchmarks. It currently supports Universal NER v2 and OpenNER 1.0.

## Datasets

| Source             | Id prefix               | Contents                                                               |
| ------------------ | ----------------------- | ---------------------------------------------------------------------- |
| [Universal NER v2] | `uner/`                 | 31 UD treebanks with PER/LOC/ORG annotations; 14 of them are test-only |
| [OpenNER 1.0]      | `openner-core_types/`   | 36 corpora in 52 languages, mapped to PER/LOC/ORG                      |
| [OpenNER 1.0]      | `openner-standardized/` | The same corpora with their original (standardized) entity types       |

[Universal NER v2]: https://www.universalner.org/
[OpenNER 1.0]: https://github.com/bltlab/open-ner

UNER ids use the UD treebank code, for example `uner/en_ewt` or `uner/nno_norne`. The
UNER files come from the `UniversalNER` GitHub org, not the Hugging Face copy, which
only has v1 and depends on a loading script. A few UNER treebanks (hr_set, sr_set,
sl_ssj) still contain `OTH` tags. These are mapped to `O` when loaded, since the
benchmark scores only PER, LOC, and ORG.

OpenNER comes from GitHub release `v0.9.0`, the version that matches the paper. Some
UNER treebanks also appear in OpenNER, with OpenNER's own splits and fixes, so don't
count them twice when you aggregate results. Each OpenNER corpus has its own license,
and some don't allow commercial use.

## Usage

Install the project and download the data into `./data`. Set `NER_EVAL_DATA` to use a
different directory, and set `GITHUB_TOKEN` if you hit the GitHub API rate limit.

    uv sync
    uv run ner-eval download                 # all sources
    uv run ner-eval download openner --variant standardized
    uv run ner-eval list

Evaluate a method on the test split of every dataset, or of specific ones:

    uv run ner-eval evaluate memorize
    uv run ner-eval evaluate memorize uner/en_ewt uner/el_gdt --json

Test-only datasets need training data from somewhere else. Use `--train-on` for
cross-lingual or cross-domain transfer:

    uv run ner-eval evaluate memorize uner/ko_pud uner/ja_pud --train-on uner/en_ewt

Scores are exact-match, span-level micro precision, recall, and F1, following
conlleval conventions.

## Methods

| Name       | Description                                                            |
| ---------- | ---------------------------------------------------------------------- |
| `memorize` | Lexicon baseline that tags the longest entity strings seen in training |

To add a method, write a class with `fit(train: list[Sentence])` and
`predict(sents: list[list[str]]) -> list[list[str]]`, then register it in
`METHODS` in `src/ner_eval/methods/__init__.py`.

## Development

    uv run pytest
    uv run black . && uv run isort .
    uv run mypy src
