# ner-eval

A harness for comparing named entity recognition methods on [Few-NERD], a large,
manually annotated English NER dataset built from Wikipedia. Few-NERD has 8 coarse and
66 fine-grained entity types. It was introduced in [Few-NERD: A Few-shot Named Entity
Recognition Dataset][paper] (Ding et al., ACL 2021).

[Few-NERD]: https://ningding97.github.io/fewnerd/
[paper]: https://aclanthology.org/2021.acl-long.248/

## Dataset

The harness uses Few-NERD's supervised setting with its official splits. Each method
is evaluated at two label granularities, which act as the dataset ids:

| Id       | Labels                        |
| -------- | ----------------------------- |
| `coarse` | 8 types, e.g. `person`        |
| `fine`   | 66 types, e.g. `person-actor` |

| Split | Sentences | Tokens    | Entities |
| ----- | --------- | --------- | -------- |
| train | 131,767   | 3,227,563 | 340,387  |
| dev   | 18,824    | 463,214   | 48,770   |
| test  | 37,648    | 921,118   | 96,902   |

The data comes from the [Hugging Face copy] (`DFKI-SLT/few-nerd`), since the original
download link no longer works. It's licensed CC BY-SA 4.0.

[Hugging Face copy]: https://huggingface.co/datasets/DFKI-SLT/few-nerd

Few-NERD uses IO tags, with no B-/I- prefixes, so adjacent entities of the same type
can't be told apart. The loader converts IO to BIO by treating each run of the same
tag as one entity. Entity counts are therefore slightly lower than the paper's.

Few-NERD's `inter` and `intra` settings aren't supported. Their test sets use entity
types that never appear in training, and they're meant for episodic few-shot
evaluation.

## Usage

Install the project and download the data into `./data/few-nerd`. Set `NER_EVAL_DATA`
to use a different directory.

    uv sync
    uv run ner-eval download
    uv run ner-eval list

Train a method on the train split and evaluate it on test (or `--split dev`):

    uv run ner-eval evaluate memorize            # coarse and fine
    uv run ner-eval evaluate memorize fine --json

Pass method options with `-o key=value`. Use `--max-eval` to score a subset of the
evaluation split, and `--out` to append results to a JSONL file:

    uv run ner-eval evaluate bert coarse -o max_train=2000 -o epochs=1 --max-eval 1000
    uv run ner-eval evaluate modernbert --out results/runs.jsonl

Scores are exact-match, span-level micro precision, recall, and F1, following
conlleval conventions.

To train and evaluate every method on both label sets, run the sweep script. On a Mac,
`caffeinate -i` keeps the machine awake for the hours it takes:

    caffeinate -i scripts/run_all.sh

Each run writes `results/<method>-<level>.jsonl` and logs training progress to
`results/logs/`. Finished runs are skipped, so rerunning resumes an interrupted sweep;
set `FORCE=1` to redo them. `RESULTS`, `METHODS`, and `LEVELS` override the output
directory, methods, and label sets. Extra arguments are passed to `evaluate`:

    RESULTS=results/smoke METHODS="bert modernbert" scripts/run_all.sh \
        -o max_train=2000 -o epochs=1 --max-eval 1000

## Methods

| Name          | Description                                                            | Coarse F1 | Fine F1 |
| ------------- | ---------------------------------------------------------------------- | --------- | ------- |
| `memorize`    | Lexicon baseline that tags the longest entity strings seen in training | 0.342     | 0.298   |
| `bert`        | Fine-tuned `google-bert/bert-base-cased`                               |           |         |
| `roberta`     | Fine-tuned `FacebookAI/roberta-base`                                   |           |         |
| `modernbert`  | Fine-tuned `answerdotai/ModernBERT-base`                               |           |         |
| `ettin`       | Fine-tuned `jhu-clsp/ettin-encoder-150m`                               |           |         |
| `transformer` | Fine-tuned encoder of your choice, set with `-o model=<hf-id>`         |           |         |

The encoder methods fine-tune a Hugging Face model with a linear tagging head. Each
word's label is predicted from its first subword. Words are joined with spaces before
tokenization rather than passed pre-split, because some BPE tokenizers (e.g.
ModernBERT's) would otherwise drop the spaces they were trained with. Training runs
on CUDA, MPS, or CPU, whichever is available, with bf16 autocast on CUDA. The main
options and their defaults are `epochs=3`, `lr=5e-5`, `batch_size=32`, `max_len=256`
(subwords; words past it are tagged `O`), `warmup=0.1`, `max_train` (subsample the
training set), `device`, and `seed=0`.

On an M4 Pro (MPS), base-size models train at about 90 sentences per second, so one
epoch on the full training set takes about 25 minutes.

DeBERTa-v3 (`microsoft/deberta-v3-base`) isn't included yet: its loss becomes NaN
after the first optimizer step, on both CPU and MPS.

To add a method, write a class with `fit(train: list[Sentence])` and
`predict(sents: list[list[str]]) -> list[list[str]]`, then register it in
`METHODS` in `src/ner_eval/methods/__init__.py`.

## Tag Inventory

This table is generated by `uv run ner-eval tags`. Mention counts are summed across
all splits, after the IO-to-BIO conversion.

| Coarse type  | Mentions | Fine types (mentions)                                                                                                                                                                                                              |
| ------------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| art          | 21,008   | broadcastprogram (3,149), film (4,283), music (5,360), other (3,036), painting (331), writtenart (4,849)                                                                                                                           |
| building     | 25,092   | airport (1,814), hospital (1,760), hotel (1,411), library (1,727), other (12,798), restaurant (1,156), sportsfacility (2,032), theater (2,394)                                                                                     |
| event        | 20,204   | attack/battle/war/militaryconflict (5,272), disaster (1,073), election (838), other (4,588), protest (746), sportsevent (7,687)                                                                                                    |
| location     | 136,351  | GPE (102,635), bodiesofwater (5,975), island (3,217), mountain (3,328), other (10,565), park (2,144), road/railway/highway/transit (8,487)                                                                                         |
| organization | 95,674   | company (19,663), education (10,766), government/governmentagency (7,432), media/newspaper (5,972), other (22,074), politicalparty (5,159), religion (3,675), showorganization (3,888), sportsleague (4,444), sportsteam (12,601)  |
| other        | 48,146   | astronomything (3,673), award (4,790), biologything (8,777), chemicalthing (5,210), currency (3,721), disease (4,065), educationaldegree (1,903), god (2,676), language (4,055), law (2,434), livingthing (4,598), medical (2,244) |
| person       | 108,557  | actor (8,167), artist/author (17,616), athlete (14,865), director (3,014), other (43,708), politician (14,055), scholar (3,676), soldier (3,456)                                                                                   |
| product      | 31,027   | airplane (3,842), car (3,638), food (2,270), game (2,238), other (8,040), ship (1,995), software (4,393), train (1,653), weapon (2,958)                                                                                            |

## Development

    uv run pytest
    uv run black . && uv run isort .
    uv run mypy src
