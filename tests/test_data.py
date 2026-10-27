from ner_eval import data


def test_list_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CONLL_DIR", tmp_path)
    assert data.list_datasets() == []

    for s in data.SPLITS:
        (tmp_path / f"cleanconll.{s}").write_text("Rome\tNNP\tO\tB-LOC\tB-LOC\n")
    assert data.list_datasets() == ["cleanconll"]
    assert set(data.load("cleanconll")) == {"train", "dev", "test"}


def test_read_conll(tmp_path):
    rows = [
        "-DOCSTART-\t-X-\tO\tO\tO",
        "",
        "Ann\tNNP\tO\tB-ORG\tB-PER",
        "in\tIN\tO\tO\tO",
    ]
    (tmp_path / "x").write_text(
        "\n".join(rows + ["", "Rome\tNNP\tO\tB-LOC\tB-LOC", ""])
    )
    a, b = data.read_conll(tmp_path / "x")
    assert (a.tokens, a.tags) == (["Ann", "in"], ["B-PER", "O"])
    assert (b.tokens, b.tags) == (["Rome"], ["B-LOC"])


def test_run_name():
    assert data.run_name("bert", "cleanconll") == "conll-bert-cleanconll"
