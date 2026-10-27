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

| Method             | F1    | F1, 50% overlap |
| ------------------ | ----- | --------------- |
| `memorize`         | 0.569 | 0.610           |
| `bert`             | 0.933 | 0.945           |
| `roberta`          | 0.941 | 0.950           |
| `roberta-large`    | 0.955 | 0.962           |
| `modernbert`       | 0.933 | 0.947           |
| `modernbert-large` | 0.945 | 0.956           |
| `deberta`          | 0.949 | 0.957           |
| `deberta-large`    | 0.953 | 0.962           |
| `ettin-17m`        | 0.863 | 0.886           |
| `ettin-32m`        | 0.900 | 0.917           |
| `ettin-68m`        | 0.908 | 0.927           |
| `ettin`            | 0.930 | 0.940           |
| `ettin-400m`       | 0.938 | 0.952           |
| `ettin-1b`         | 0.946 | 0.958           |
| `ettin-dec-17m`    | 0.743 | 0.798           |
| `ettin-dec-32m`    | 0.777 | 0.827           |
| `ettin-dec-68m`    | 0.811 | 0.857           |
| `ettin-dec-150m`   | 0.820 | 0.866           |
| `ettin-dec-400m`   | 0.837 | 0.881           |
| `ettin-dec-1b`     | 0.844 | 0.886           |
| `qwen3-0.6b`       | 0.829 | 0.873           |
| `qwen3-1.7b`       | 0.851 | 0.894           |

Scores are span-level micro F1 on the test split with the default options and one seed.
The 50% overlap column also counts a predicted span as correct if its token overlap
with a gold span of the same type is at least 0.5 (`--overlap 0.5`). The whole sweep
takes under two hours on an NVIDIA GB10.

The same models score 0.93 to 0.955 here but about 0.79 on Few-NERD's 8 coarse types.
CleanCoNLL is an easier task, with 4 types and news text, but cleaner labels likely
account for part of the gap. Accepting partial matches adds only about 0.01 F1 for the
encoders, against about 0.055 on Few-NERD, so span boundaries are much more consistent
here. Decoders still trail the encoders by about 0.1 F1.

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
