from collections import Counter

from ner_eval.conll_diff import changes, diff_doc


def test_changes():
    orig = [(0, 1, "LOC"), (3, 4, "ORG"), (6, 7, "PER")]
    clean = [(0, 1, "ORG"), (3, 5, "ORG"), (8, 9, "MISC")]
    assert changes(orig, clean) == [
        ["type", (0, 1, "LOC"), (0, 1, "ORG")],
        ["boundary", (3, 4, "ORG"), (3, 5, "ORG")],
        ["removed", (6, 7, "PER"), None],
        ["added", None, (8, 9, "MISC")],
    ]


def test_diff_doc_aligns_resplit_sentences():
    # CleanCoNLL merges two sentences and relabels JAPAN as a team
    od = [[("JAPAN", "I-LOC"), ("WIN", "O")], [("Tokyo", "I-LOC")]]
    cd = [[("JAPAN", "B-ORG"), ("WIN", "O"), ("Tokyo", "B-LOC")]]
    stats = Counter()
    [sent] = diff_doc(od, cd, stats)
    assert sent["changes"] == [["type", (0, 1, "LOC"), (0, 1, "ORG")]]
    assert stats == Counter(sentences=1, entities_orig=2, entities_clean=2)
