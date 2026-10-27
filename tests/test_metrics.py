from ner_eval.metrics import iou, score, spans


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


def test_iou():
    assert iou((0, 2, "PER"), (0, 2, "LOC")) == 1.0
    assert iou((0, 2, "PER"), (1, 4, "PER")) == 0.25
    assert iou((0, 2, "PER"), (2, 4, "PER")) == 0.0


def test_score_overlap():
    gold = [["B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]]
    pred = [["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG"]]
    res = score(gold, pred, overlap=0.5)
    assert (res["ALL"].tp, res["ALL"].fp, res["ALL"].fn) == (1, 1, 1)
    assert score(gold, pred, overlap=0.7)["ALL"].tp == 0


def test_score_overlap_matches_one_to_one():
    # both halves overlap the gold span by exactly 0.5, but only one can claim it
    res = score([["B-PER", "I-PER"]], [["B-PER", "B-PER"]], overlap=0.5)
    assert (res["ALL"].tp, res["ALL"].fp, res["ALL"].fn) == (1, 1, 0)
