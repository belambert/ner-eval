from ner_eval.metrics import score, spans


def test_spans():
    assert spans(["B-PER", "I-PER", "O", "B-LOC", "B-LOC"]) == [
        (0, 2, "PER"),
        (3, 4, "LOC"),
        (4, 5, "LOC"),
    ]


def test_spans_stray_and_mismatched_inside():
    assert spans(["I-ORG", "I-ORG", "I-PER"]) == [(0, 2, "ORG"), (2, 3, "PER")]


def test_spans_hyphenated_types():
    assert spans(["B-ART-DERIV", "I-ART-DERIV"]) == [(0, 2, "ART-DERIV")]


def test_score():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    pred = [["B-PER", "I-PER", "O", "B-ORG"]]
    res = score(gold, pred)
    assert (res["ALL"].tp, res["ALL"].fp, res["ALL"].fn) == (1, 1, 1)
    assert res["ALL"].f1 == 0.5
    assert res["PER"].f1 == 1.0
    assert res["LOC"].r == 0.0 and res["ORG"].p == 0.0


def test_score_partial_match_is_wrong():
    res = score([["B-PER", "I-PER"]], [["B-PER", "O"]])
    assert res["ALL"].tp == 0
