from ner_eval.inventory import md_table


def test_md_table_aligns_columns():
    assert md_table(["A", "Bee"], [["xyz", "1"]]) == (
        "| A   | Bee |\n| --- | --- |\n| xyz | 1   |"
    )
