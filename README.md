# ner-eval

A harness for training and comparing named entity recognition methods, fine-tuned
encoders and decoders, on English NER datasets. Each method is trained on a dataset's
train split and scored on its test split.

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

Train a method on the train split, save it to `results/models/` (or `--models DIR`),
and score it on test (or `--split dev`). Each model is saved in
`<corpus>-<method>-<dataset>`, e.g. `results/models/conll-bert-cleanconll`:

    uv run ner-eval train bert            # every downloaded dataset
    uv run ner-eval train bert cleanconll --json

Pass method options with `-o key=value`. Use `--max-eval` to score a subset of the
evaluation split, and `--out` to append results to a JSONL file:

    uv run ner-eval train bert cleanconll -o max_train=2000 -o epochs=1 --max-eval 1000
    uv run ner-eval train modernbert --out results/runs.jsonl

`evaluate` scores saved models without retraining. It takes the same scoring options
as `train`, and `--models` can also be a Hugging Face user or org that the models were
pushed to:

    uv run ner-eval evaluate bert cleanconll
    uv run ner-eval evaluate bert cleanconll --split dev --models <hf-user>

Scores are exact-match, span-level micro precision, recall, and F1, following
conlleval conventions. Use `--overlap` to also accept partial matches. A predicted
span then counts as correct if its type matches a gold span and its token overlap
with that span (intersection over union) is at least the given fraction. Each gold
span can be matched only once. The default of 1.0 is exact match:

    uv run ner-eval evaluate bert --overlap 0.5

To rescore without rerunning the model at all, save its predictions with `--preds DIR`.
This writes `DIR/<corpus>-<method>-<dataset>-<split>.jsonl`, one JSON list of predicted
tags per sentence, in split order. `rescore` reads the dataset and split from the file
name, loads the gold tags, and prints the same scores as `evaluate` (`--json` too):

    uv run ner-eval evaluate bert --preds results/preds
    uv run ner-eval rescore results/preds/*.jsonl --overlap 0.5

The separate `ner-errors` command turns a predictions file into an HTML page of every
sentence the model got wrong, showing gold and predicted entities side by side. It
counts wrong types, boundary errors, missed entities, and spurious ones, with filters
by error kind and type:

    uv run ner-errors results/preds/conll-roberta-large-cleanconll-test.jsonl site/errors.html

The page for `roberta-large` on CleanCoNLL is published on [GitHub Pages][errors].
So is a [page of every label CleanCoNLL changed from CoNLL-03][diff], written by
`conll-diff` (see [docs/cleanconll.md](docs/cleanconll.md)).

[errors]: https://belambert.github.io/ner-eval/roberta-large-errors.html
[diff]: https://belambert.github.io/ner-eval/

To train and evaluate every method on every dataset, run the sweep script:

    scripts/run_all.sh

Each run saves its model to `results/models/`, writes `results/<method>-<dataset>.jsonl`,
saves its predictions to `results/preds/` for `rescore`, and logs training progress to
`results/logs/`. Finished runs are skipped, so rerunning resumes an interrupted sweep;
set `FORCE=1` to redo them. `RESULTS`, `METHODS`, and `DATASETS` override the output
directory, methods, and datasets. Extra arguments are passed to `train`:

    RESULTS=results/smoke METHODS="bert modernbert" DATASETS=cleanconll scripts/run_all.sh \
        -o max_train=2000 -o epochs=1 --max-eval 1000

To push the saved transformer models to private Hugging Face repos:

    for d in results/models/*/; do
        [[ -f $d/config.json ]] && uv run hf upload "<hf-user>/$(basename $d)" "$d" . --private
    done

## Methods

| Name                | Description                                                            | Params |
| ------------------- | ---------------------------------------------------------------------- | ------ |
| `bert`              | Fine-tuned `google-bert/bert-base-cased`                               | 108M   |
| `roberta`           | Fine-tuned `FacebookAI/roberta-base`                                   | 124M   |
| `roberta-large`     | Fine-tuned `FacebookAI/roberta-large`                                  | 354M   |
| `xlm-roberta-large` | Fine-tuned `FacebookAI/xlm-roberta-large`                              | 560M   |
| `modernbert`        | Fine-tuned `answerdotai/ModernBERT-base`                               | 150M   |
| `modernbert-large`  | Fine-tuned `answerdotai/ModernBERT-large`                              | 396M   |
| `deberta`           | Fine-tuned `microsoft/deberta-v3-base`                                 | 184M   |
| `deberta-large`     | Fine-tuned `microsoft/deberta-v3-large`                                | 434M   |
| `qwen3-0.6b`        | Fine-tuned decoder `Qwen/Qwen3-0.6B`                                   | 596M   |
| `qwen3-1.7b`        | Fine-tuned decoder `Qwen/Qwen3-1.7B`                                   | 1.72B  |
| `qwen3-4b`          | Fine-tuned decoder `Qwen/Qwen3-4B`                                     | 4.02B  |

Params counts the fine-tuned model, including the tagging head, whose size varies
slightly by dataset. Results are in each dataset's doc. The Ettin models were removed
to shorten the list; their results are in [docs/ettin.md](docs/ettin.md).

The transformer methods fine-tune a Hugging Face model with a linear tagging head.
Encoders predict each word's label from its first subword. Decoders use its last
subword (`subword=last`), since causal attention only lets a subword see the text
before it. By default their attention stays causal, so a word's label can't depend on
the words after it. `bidirectional=True` drops the causal mask during fine-tuning and
prediction, so every subword attends to the whole sentence, as in an encoder.
`repeat=True` instead feeds each sentence twice and labels only the second copy, so
every labeled subword can attend to the whole sentence through the first copy. It
needs no change to the model but doubles the sequence length (up to twice `max_len`).
The Qwen3 models use `lr=2e-5`, a more typical rate for full fine-tuning of
LLMs. Words are joined with spaces before tokenization rather than passed pre-split,
because some BPE tokenizers (e.g. ModernBERT's) would otherwise drop the spaces they
were trained with. Training runs on CUDA, MPS, or CPU, whichever is available, with
fp32 weights (whatever the checkpoint's dtype) and bf16 autocast on CUDA.
`checkpointing=True` recomputes activations during the backward pass, which trains
the same model with much less memory but more slowly. `prompt` puts unlabeled text
before each sentence, as the first of a sentence pair, and `max_len` still counts only
the sentence. With `repeat=True`, only the sentence is repeated. `prompts/` has a
summary of the CleanCoNLL guidelines and a filler prompt of the same length:

    uv run ner-eval train qwen3-0.6b cleanconll -o prompt="$(cat prompts/cleanconll.txt)"

The main options and their defaults are `epochs=3`, `lr=5e-5`, `batch_size=32`,
`max_len=256` (subwords; words past it are tagged `O`), `warmup=0.1`, `max_train`
(subsample the training set), `subword=first`, `bidirectional=False`, `repeat=False`,
`checkpointing=False`, `prompt=""`, `device`, and `seed=0`.

To add a method, write a class with `fit(train: list[Sentence])` and
`predict(sents: list[list[str]]) -> list[list[str]]`, then register it in
`METHODS` in `src/ner_eval/methods/__init__.py`.

## Development

    uv run pytest
    uv run black . && uv run isort .
    uv run mypy src
