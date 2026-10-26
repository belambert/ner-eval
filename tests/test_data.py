import json

import pyarrow as pa
import pyarrow.parquet as pq

from ner_eval import data


def write_parquet(path, rows):
    """Write a parquet file shaped like the Hugging Face Few-NERD export."""
    labels = lambda names: {"feature": {"names": names, "_type": "ClassLabel"}}
    features = {
        "ner_tags": labels(["O", "location", "person"]),
        "fine_ner_tags": labels(["O", "location-GPE", "person-actor"]),
    }
    meta = {b"huggingface": json.dumps({"info": {"features": features}})}
    table = pa.Table.from_pylist(rows).replace_schema_metadata(meta)
    pq.write_table(table, path)


def test_io_to_bio():
    assert data.io_to_bio(["person", "person", "O", "location", "person"]) == [
        "B-person",
        "I-person",
        "O",
        "B-location",
        "B-person",
    ]


def test_read_parquet(tmp_path):
    row = {"tokens": ["Ann", "Lee", "in", "Rome"], "ner_tags": [2, 2, 0, 1]}
    row["fine_ner_tags"] = row["ner_tags"]
    write_parquet(tmp_path / "x.parquet", [row])

    [coarse] = data.read_parquet(tmp_path / "x.parquet", "coarse")
    assert coarse.tokens == ["Ann", "Lee", "in", "Rome"]
    assert coarse.tags == ["B-person", "I-person", "O", "B-location"]
    [fine] = data.read_parquet(tmp_path / "x.parquet", "fine")
    assert fine.tags == ["B-person-actor", "I-person-actor", "O", "B-location-GPE"]


def test_list_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    assert data.list_datasets() == []

    row = {"tokens": ["Rome"], "ner_tags": [1], "fine_ner_tags": [1]}
    for s in data.SPLITS:
        write_parquet(tmp_path / f"{s}.parquet", [row])
    assert data.list_datasets() == ["coarse", "fine"]
    assert set(data.load("fine")) == {"train", "dev", "test"}
