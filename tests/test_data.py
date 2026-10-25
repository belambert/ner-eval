from ner_eval import data


def test_read_uner(tmp_path):
    p = tmp_path / "xx_tb-ud-dev.iob2"
    p.write_text(
        "# sent_id = 1\n"
        "# text = Ana in Zagreb\n"
        "1\tAna\tB-PER\t-\tx\n"
        "2\tin\tO\t-\t-\n"
        "3\tZagreb\tB-OTH\t-\tx\n"
        "\n"
        "1\tHi\tO\t-\t-\n"
    )
    sents = data.read_uner(p)
    assert [s.tokens for s in sents] == [["Ana", "in", "Zagreb"], ["Hi"]]
    assert sents[0].tags == ["B-PER", "O", "O"]


def test_read_conll(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("Ana B-PER\nin O\n\n\nRim B-LOC\n")
    sents = data.read_conll(p)
    assert [s.tags for s in sents] == [["B-PER", "O"], ["B-LOC"]]


def test_list_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    uner = tmp_path / "uner"
    uner.mkdir()
    for f in ("xx_tb-ud-train.iob2", "xx_tb-ud-test.iob2", "yy_legal.iob2"):
        (uner / f).write_text("1\tA\tO\t-\t-\n")
    onr = tmp_path / "openner" / "core_types" / "Corp" / "xxx"
    onr.mkdir(parents=True)
    (onr / "test.txt").write_text("A O\n")

    assert data.list_datasets() == [
        "uner/xx_tb",
        "uner/yy_legal",
        "openner-core_types/Corp/xxx",
    ]
    assert set(data.load("uner/xx_tb")) == {"train", "test"}
    assert set(data.load("uner/yy_legal")) == {"test"}
    assert set(data.load("openner-core_types/Corp/xxx")) == {"test"}
