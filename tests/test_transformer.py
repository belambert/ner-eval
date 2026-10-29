import pytest
import torch
from transformers import Qwen3Config, Qwen3ForTokenClassification

from ner_eval.cli import _parse_opt
from ner_eval.methods.transformer import Transformer, make_bidirectional


class Enc(dict):
    """A dict with the `sequence_ids` method of a tokenizer's BatchEncoding."""

    def sequence_ids(self):
        return self.seq


class FakeTok:
    """Splits on spaces and in half for words over 4 chars, with BPE-style offsets.

    A pair is encoded as [CLS] a [SEP] b [SEP].
    """

    def __call__(self, *texts, add_special_tokens=True, **_):
        offs, seq = [], []
        for q, text in enumerate(texts):
            o = self._offsets(text)
            offs += o + [(0, 0)] * add_special_tokens
            seq += [q] * len(o) + [None] * add_special_tokens
        if add_special_tokens:
            offs, seq = [(0, 0)] + offs, [None] + seq
        enc = Enc(input_ids=list(range(len(offs))), offset_mapping=offs)
        enc.seq = seq
        return enc

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def _offsets(self, text):
        offs, i = [], 0
        for w in text.split(" "):
            # offsets after the first word include the leading space, as in ModernBERT
            start = i - 1 if i else i
            if len(w) > 4:
                offs += [(start, i + len(w) // 2), (i + len(w) // 2, i + len(w))]
            else:
                offs.append((start, i + len(w)))
            i += len(w) + 1
        return offs


def test_encode_marks_first_subwords():
    t = Transformer(max_len=512)
    t.tok = FakeTok()
    ids, wids = t._encode(["Ann", "visited", "Rome"])
    assert len(ids) == 6
    assert wids == [None, 0, 1, None, 2, None]


def test_encode_marks_last_subwords():
    t = Transformer(max_len=512, subword="last")
    t.tok = FakeTok()
    _, wids = t._encode(["Ann", "visited", "Rome"])
    assert wids == [None, 0, None, 1, 2, None]


def test_encode_repeat_labels_second_copy():
    t = Transformer(max_len=512, subword="last", repeat=True)
    t.tok = FakeTok()
    ids, wids = t._encode(["Ann", "visited", "Rome"])
    assert ids == list(range(6)) * 2
    assert wids == [None] * 6 + [None, 0, None, 1, 2, None]


def test_encode_prompt_is_unlabeled():
    t = Transformer(max_len=512, prompt="Tag it")
    t.tok = FakeTok()
    ids, wids = t._encode(["Ann", "visited", "Rome"])
    # [CLS] Tag it [SEP] Ann vis ited Rome [SEP]
    assert len(ids) == 9
    assert wids == [None] * 4 + [0, 1, None, 2, None]


def test_encode_repeat_with_prompt_repeats_only_sentence():
    t = Transformer(max_len=512, prompt="Tag it", repeat=True)
    t.tok = FakeTok()
    ids, wids = t._encode(["Ann", "visited", "Rome"])
    assert ids == list(range(9)) + list(range(4, 9))
    assert wids == [None] * 9 + [0, 1, None, 2, None]


def test_parse_opt():
    assert _parse_opt("epochs=2") == ("epochs", 2)
    assert _parse_opt("lr=3e-5") == ("lr", 3e-5)
    assert _parse_opt("model=google-bert/bert-base-cased") == (
        "model",
        "google-bert/bert-base-cased",
    )


@pytest.mark.parametrize("bidirectional", [False, True])
def test_bidirectional_sees_later_tokens(bidirectional):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=50,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_labels=3,
    )
    t = Transformer(device="cpu", bidirectional=bidirectional)
    t.dev, t.net = torch.device("cpu"), Qwen3ForTokenClassification(config).eval()
    if bidirectional:
        make_bidirectional(t.net)

    # the second row changes only the last token, and pads the third row's end
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]])
    with torch.no_grad():
        logits = t._run(ids, mask).logits
    first_changed = not torch.allclose(logits[0, 0], logits[1, 0])
    assert first_changed == bidirectional
    # padding is masked either way, so a padded row matches the unpadded prefix
    short = t._run(ids[:1, :3], mask[:1, :3]).logits
    assert torch.allclose(logits[2, :3], short[0], atol=1e-5)
