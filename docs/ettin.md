# Ettin

[Ettin] (Weller et al., 2025) is a suite of paired encoder and decoder models, from 17M
to 1B parameters, trained on the same data with the same recipe. That makes it a clean
way to compare encoders and decoders at the same size. The Ettin methods were removed
from the registry to shorten the method list; this page keeps their results.

[Ettin]: https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb

## Models

| Method           | Model                         | Params |
| ---------------- | ----------------------------- | ------ |
| `ettin-17m`      | `jhu-clsp/ettin-encoder-17m`  | 17M    |
| `ettin-32m`      | `jhu-clsp/ettin-encoder-32m`  | 32M    |
| `ettin-68m`      | `jhu-clsp/ettin-encoder-68m`  | 68M    |
| `ettin`          | `jhu-clsp/ettin-encoder-150m` | 150M   |
| `ettin-400m`     | `jhu-clsp/ettin-encoder-400m` | 396M   |
| `ettin-1b`       | `jhu-clsp/ettin-encoder-1b`   | 1.03B  |
| `ettin-dec-17m`  | `jhu-clsp/ettin-decoder-17m`  | 17M    |
| `ettin-dec-32m`  | `jhu-clsp/ettin-decoder-32m`  | 32M    |
| `ettin-dec-68m`  | `jhu-clsp/ettin-decoder-68m`  | 68M    |
| `ettin-dec-150m` | `jhu-clsp/ettin-decoder-150m` | 149M   |
| `ettin-dec-400m` | `jhu-clsp/ettin-decoder-400m` | 395M   |
| `ettin-dec-1b`   | `jhu-clsp/ettin-decoder-1b`   | 1.03B  |

The `transformer` method can still run any of them. Decoders need `subword=last`:

    uv run ner-eval train transformer cleanconll -o model=jhu-clsp/ettin-encoder-150m
    uv run ner-eval train transformer cleanconll -o model=jhu-clsp/ettin-decoder-150m -o subword=last

## Results

Span-level micro F1 on each test split with the default options. The 50% columns use
`--overlap 0.5`. See [cleanconll.md](cleanconll.md) and [few-nerd.md](few-nerd.md) for
the other methods and how the scores were computed.

| Method           | CleanCoNLL | CleanCoNLL 50% | Few-NERD coarse | Few-NERD fine | Coarse 50% | Fine 50% |
| ---------------- | ---------- | -------------- | --------------- | ------------- | ---------- | -------- |
| `ettin-17m`      | 0.863      | 0.886          | 0.749           | 0.604         | 0.815      | 0.675    |
| `ettin-32m`      | 0.900      | 0.917          | 0.765           | 0.632         | 0.825      | 0.699    |
| `ettin-68m`      | 0.908      | 0.927          | 0.776           | 0.651         | 0.834      | 0.712    |
| `ettin`          | 0.930      | 0.940          | 0.779           | 0.659         | 0.839      | 0.720    |
| `ettin-400m`     | 0.938      | 0.952          | 0.785           | 0.669         | 0.842      | 0.729    |
| `ettin-1b`       | 0.946      | 0.958          | 0.789           | 0.673         | 0.845      | 0.731    |
| `ettin-dec-17m`  | 0.743      | 0.798          | 0.621           | 0.443         | 0.721      | 0.551    |
| `ettin-dec-32m`  | 0.777      | 0.827          | 0.642           | 0.474         | 0.738      | 0.577    |
| `ettin-dec-68m`  | 0.811      | 0.857          | 0.665           | 0.508         | 0.757      | 0.606    |
| `ettin-dec-150m` | 0.820      | 0.866          | 0.671           | 0.518         | 0.762      | 0.616    |
| `ettin-dec-400m` | 0.837      | 0.881          | 0.684           | 0.535         | 0.772      | 0.631    |
| `ettin-dec-1b`   | 0.844      | 0.886          | 0.689           | –             | 0.775      | –        |

At every size, the encoder beats its matching decoder by about 0.1 F1 on CleanCoNLL
and Few-NERD coarse, and by about 0.15 on Few-NERD fine. The gap narrows only slightly
with scale. Partial matches help the decoders more, since causal attention keeps them
from seeing the words after an entity, so they make more boundary errors. `ettin-1b`
comes within about 0.01 F1 of the best encoders on both datasets.
