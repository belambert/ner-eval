# ner-eval

A harness for training and comparing named entity recognition methods, from a lexicon
baseline to fine-tuned encoders and decoders, on English NER datasets. Each method is
trained on a dataset's train split and scored on its test split.

## Datasets

| Id           | Corpus     | Types                          | Train sentences | Details                                  |
| ------------ | ---------- | ------------------------------ | --------------- | ---------------------------------------- |
| `cleanconll` | CleanCoNLL | 4: `PER`, `LOC`, `ORG`, `MISC` | 13,957          | [docs/cleanconll.md](docs/cleanconll.md) |

Each doc describes the dataset and has its results and analysis. The harness used to
support [Few-NERD]; its code was removed, but [docs/few-nerd.md](docs/few-nerd.md) keeps
its results and error analysis.

[Few-NERD]: https://ningding97.github.io/fewnerd/

## Usage

Install the project and download the data into `./data` (or set `NER_EVAL_DATA`):

    uv sync
    uv run ner-eval download
    uv run ner-eval list

Train a method on the train split and evaluate it on test (or `--split dev`):

    uv run ner-eval evaluate memorize            # every downloaded dataset
    uv run ner-eval evaluate memorize cleanconll --json

Pass method options with `-o key=value`. Use `--max-eval` to score a subset of the
evaluation split, and `--out` to append results to a JSONL file:

    uv run ner-eval evaluate bert cleanconll -o max_train=2000 -o epochs=1 --max-eval 1000
    uv run ner-eval evaluate modernbert --out results/runs.jsonl

Use `--save DIR` to keep each trained model in `DIR/<corpus>-<method>-<dataset>` (e.g.
`conll-bert-cleanconll`), and
`--load DIR` to evaluate saved models without retraining. For transformer methods,
`DIR` can also be a Hugging Face user or org that the models were pushed to:

    uv run ner-eval evaluate bert cleanconll --save results/models
    uv run ner-eval evaluate bert cleanconll --load results/models
    uv run ner-eval evaluate bert cleanconll --load <hf-user>

Scores are exact-match, span-level micro precision, recall, and F1, following
conlleval conventions. Use `--overlap` to also accept partial matches. A predicted
span then counts as correct if its type matches a gold span and its token overlap
with that span (intersection over union) is at least the given fraction. Each gold
span can be matched only once. The default of 1.0 is exact match. Combined with
`--load`, this rescores saved models without retraining:

    uv run ner-eval evaluate memorize --overlap 0.5
    uv run ner-eval evaluate bert --load <hf-user> --overlap 0.5

To rescore without rerunning the model at all, save its predictions with `--preds DIR`.
This writes `DIR/<corpus>-<method>-<dataset>-<split>.jsonl`, one JSON list of predicted
tags per sentence, in split order. `rescore` reads the dataset and split from the file
name, loads the gold tags, and prints the same scores as `evaluate` (`--json` too):

    uv run ner-eval evaluate memorize --preds results/preds
    uv run ner-eval rescore results/preds/*.jsonl --overlap 0.5

To train and evaluate every method on every dataset, run the sweep script:

    scripts/run_all.sh

Each run writes `results/<method>-<dataset>.jsonl`, saves its predictions to
`results/preds/` for `rescore`, and logs training progress to `results/logs/`. Finished
runs are skipped, so rerunning resumes an interrupted sweep; set `FORCE=1` to redo them.
Set `SAVE=1` to also save each model to `results/models/`. `RESULTS`, `METHODS`, and
`DATASETS` override the output directory, methods, and datasets. Extra arguments are
passed to `evaluate`:

    RESULTS=results/smoke METHODS="bert modernbert" DATASETS=cleanconll scripts/run_all.sh \
        -o max_train=2000 -o epochs=1 --max-eval 1000

To push the saved transformer models to private Hugging Face repos:

    for d in results/models/*/; do
        [[ -f $d/config.json ]] && uv run hf upload "<hf-user>/$(basename $d)" "$d" . --private
    done

## Methods

| Name               | Description                                                            | Params |
| ------------------ | ---------------------------------------------------------------------- | ------ |
| `memorize`         | Lexicon baseline that tags the longest entity strings seen in training | –      |
| `bert`             | Fine-tuned `google-bert/bert-base-cased`                               | 108M   |
| `roberta`          | Fine-tuned `FacebookAI/roberta-base`                                   | 124M   |
| `roberta-large`    | Fine-tuned `FacebookAI/roberta-large`                                  | 354M   |
| `modernbert`       | Fine-tuned `answerdotai/ModernBERT-base`                               | 150M   |
| `modernbert-large` | Fine-tuned `answerdotai/ModernBERT-large`                              | 396M   |
| `deberta`          | Fine-tuned `microsoft/deberta-v3-base`                                 | 184M   |
| `deberta-large`    | Fine-tuned `microsoft/deberta-v3-large`                                | 434M   |
| `ettin`            | Fine-tuned `jhu-clsp/ettin-encoder-150m`                               | 150M   |
| `ettin-400m`       | Fine-tuned `jhu-clsp/ettin-encoder-400m`                               | 396M   |
| `ettin-1b`         | Fine-tuned `jhu-clsp/ettin-encoder-1b`                                 | 1.03B  |
| `ettin-dec-150m`   | Fine-tuned decoder `jhu-clsp/ettin-decoder-150m`                       | 149M   |
| `ettin-dec-400m`   | Fine-tuned decoder `jhu-clsp/ettin-decoder-400m`                       | 395M   |
| `ettin-dec-1b`     | Fine-tuned decoder `jhu-clsp/ettin-decoder-1b`                         | 1.03B  |
| `qwen3-0.6b`       | Fine-tuned decoder `Qwen/Qwen3-0.6B`                                   | 596M   |
| `qwen3-1.7b`       | Fine-tuned decoder `Qwen/Qwen3-1.7B`                                   | 1.72B  |
| `transformer`      | Fine-tuned model of your choice, set with `-o model=<hf-id>`           |        |

Params counts the fine-tuned model, including the tagging head, whose size varies
slightly by dataset. Results are in each dataset's doc.

The transformer methods fine-tune a Hugging Face model with a linear tagging head.
Encoders predict each word's label from its first subword. Decoders use its last
subword (`subword=last`), since causal attention only lets a subword see the text
before it. Their attention stays causal, so a word's label can't depend on the words
after it. The Qwen3 models use `lr=2e-5`, a more typical rate for full fine-tuning of
LLMs. Words are joined with spaces before tokenization rather than passed pre-split,
because some BPE tokenizers (e.g. ModernBERT's) would otherwise drop the spaces they
were trained with. Training runs on CUDA, MPS, or CPU, whichever is available, with
fp32 weights (whatever the checkpoint's dtype) and bf16 autocast on CUDA. The main
options and their defaults are `epochs=3`, `lr=5e-5`, `batch_size=32`, `max_len=256`
(subwords; words past it are tagged `O`), `warmup=0.1`, `max_train` (subsample the
training set), `subword=first`, `device`, and `seed=0`.

To add a method, write a class with `fit(train: list[Sentence])` and
`predict(sents: list[list[str]]) -> list[list[str]]`, then register it in
`METHODS` in `src/ner_eval/methods/__init__.py`.

## Development

    uv run pytest
    uv run black . && uv run isort .
    uv run mypy src
