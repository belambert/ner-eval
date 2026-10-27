from ner_eval.cli import _parse_opt
from ner_eval.methods.transformer import Transformer


class FakeTok:
    """Splits on spaces and in half for words over 4 chars, with BPE-style offsets."""

    def __call__(self, text, **_):
        offs, i = [(0, 0)], 0
        for w in text.split(" "):
            # offsets after the first word include the leading space, as in ModernBERT
            start = i - 1 if i else i
            if len(w) > 4:
                offs += [(start, i + len(w) // 2), (i + len(w) // 2, i + len(w))]
            else:
                offs.append((start, i + len(w)))
            i += len(w) + 1
        offs.append((0, 0))
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


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


def test_parse_opt():
    assert _parse_opt("epochs=2") == ("epochs", 2)
    assert _parse_opt("lr=3e-5") == ("lr", 3e-5)
    assert _parse_opt("model=google-bert/bert-base-cased") == (
        "model",
        "google-bert/bert-base-cased",
    )
