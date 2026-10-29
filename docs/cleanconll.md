# CleanCoNLL

[CleanCoNLL] ([Rücker and Akbik, EMNLP 2023][paper]) relabels the CoNLL-03 English news
corpus, updating about 7% of its labels, with types `PER`, `LOC`, `ORG`, and `MISC` in
BIO. Its dataset id is `cleanconll`.

[CleanCoNLL]: https://github.com/flairNLP/CleanCoNLL
[paper]: https://aclanthology.org/2023.emnlp-main.533/

## Dataset

CleanCoNLL only publishes its labels, so `download` builds the data with the
release's script, which fetches the original CoNLL-03 text and patches it. This needs
`git`, `wget`, `unzip`, and `patch`. CleanCoNLL also fixes some sentence splits and
tokens, so its counts differ slightly from CoNLL-03's.

| Split | Sentences | Tokens  | Entities |
| ----- | --------- | ------- | -------- |
| train | 13,957    | 203,657 | 23,566   |
| dev   | 3,233     | 51,383  | 5,966    |
| test  | 3,427     | 46,504  | 5,725    |

## Results

| Method              | F1    | F1, 50% overlap |
| ------------------- | ----- | --------------- |
| `bert`              | 0.933 | 0.945           |
| `roberta`           | 0.941 | 0.950           |
| `roberta-large`     | 0.955 | 0.962           |
| `xlm-roberta-large` | 0.949 | 0.956           |
| `modernbert`        | 0.933 | 0.947           |
| `modernbert-large`  | 0.945 | 0.956           |
| `deberta`           | 0.949 | 0.957           |
| `deberta-large`     | 0.953 | 0.962           |
| `qwen3-0.6b`        | 0.830 | 0.871           |
| `qwen3-1.7b`        | 0.851 | 0.893           |
| `qwen3-4b`          | 0.851 | 0.890           |

Scores are span-level micro F1 on the test split with the default options and one seed.
The 50% overlap column also counts a predicted span as correct if its token overlap
with a gold span of the same type is at least 0.5 (`--overlap 0.5`). The whole sweep
takes about two hours on an NVIDIA GB10, nearly an hour of it for `qwen3-4b`. The
encoders reproduce the previous sweep's scores exactly, and the Qwen3 models' scores
move by at most 0.002.

The same models score 0.93 to 0.955 here but about 0.79 on Few-NERD's 8 coarse types.
CleanCoNLL is an easier task, with 4 types and news text, but cleaner labels likely
account for part of the gap. Accepting partial matches adds only about 0.01 F1 for the
encoders, against about 0.055 on Few-NERD, so span boundaries are much more consistent
here. Decoders still trail the encoders by about 0.1 F1, and `qwen3-4b` scores no
better than `qwen3-1.7b`, so scaling the decoder doesn't close the gap, but letting
them see the whole sentence does (see [below](#seeing-the-whole-sentence)).
The [paper]'s best results use `xlm-roberta-large`, reaching 97.08 F1 with a biaffine
span classifier and 96.98 with FLERT's document context. Here, with a plain linear
tagger on single sentences, it scores 0.949, below `roberta-large`, so the gap to those
results likely comes from the span classifier and the context, not the encoder.

[roberta-large-errors.html](roberta-large-errors.html) shows every sentence in the test
split where `roberta-large` makes an error, written by `ner-errors`. Of its 325 errors,
128 are wrong types, 108 are boundary errors, 56 are spurious entities, and 33 are
missed entities. GitHub shows the file as source, so download it to view it.

## Seeing the Whole Sentence

A decoder's causal attention keeps each word from seeing the words after it. Two
options fix that: `repeat=True` feeds each sentence twice and labels the second copy,
and `bidirectional=True` drops the causal mask. Encoders already see the whole
sentence, so only `repeat` applies to them.

| Method              | Default | `repeat` | `bidirectional` | Both  |
| ------------------- | ------- | -------- | --------------- | ----- |
| `bert`              | 0.933   | 0.926    | –               | –     |
| `roberta`           | 0.941   | 0.939    | –               | –     |
| `roberta-large`     | 0.955   | 0.952    | –               | –     |
| `xlm-roberta-large` | 0.949   | 0.948    | –               | –     |
| `modernbert`        | 0.933   | 0.934    | –               | –     |
| `modernbert-large`  | 0.945   | 0.943    | –               | –     |
| `deberta`           | 0.949   | 0.950    | –               | –     |
| `deberta-large`     | 0.953   | 0.951    | –               | –     |
| `qwen3-0.6b`        | 0.830   | 0.935    | 0.925           | 0.925 |
| `qwen3-1.7b`        | 0.851   | 0.949    | 0.942           | 0.943 |
| `qwen3-4b`          | 0.851   | 0.954    | 0.946           | 0.945 |

Scores are exact-match F1 on the test split with one seed. The `qwen3-4b` runs with the
new options used `checkpointing=True` to fit in memory, which doesn't change results.

Both options add about 0.1 F1 to every decoder, and the gain now grows with size:
`qwen3-4b` with `repeat` scores 0.954, level with the best encoders. `repeat` beats
`bidirectional` by about 0.01 at every size, perhaps because the model keeps the causal
attention it was pretrained with. Combining them adds nothing over `bidirectional`
alone. For encoders, `repeat` changes F1 by -0.007 to +0.001, within the noise of a
single seed: a rerun of `qwen3-0.6b` with `bidirectional` scored 0.918 rather than 0.925.

## Changes From CoNLL-03

The separate `conll-diff` command writes an HTML page of every entity label CleanCoNLL changed from CoNLL-03,
shown sentence by sentence with filters by split, kind of change, and type. It needs
`download`, which keeps the original CoNLL-03 labels alongside CleanCoNLL's.
The `pages` workflow can build it and publish it to GitHub Pages, but only runs when
triggered manually, since the page includes CoNLL-03's Reuters text.

    uv run conll-diff site/index.html

Of 2,891 changed entities, 1,861 change only type, mostly `LOC` to `ORG` (1,275), as
when country or city names stand for sports teams. Another 642 change boundaries, 340
are added, and 48 are removed.
