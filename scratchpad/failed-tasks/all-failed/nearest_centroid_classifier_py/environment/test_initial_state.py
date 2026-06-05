import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"
TEST_SET_PATH = "/app/test_set.json"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data directory {LANCEDB_DIR} does not exist."


def test_zealt_run_id_env_set():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID environment variable must be set."


def test_training_table_exists_and_schema():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(LANCEDB_DIR)
    name = f"train_data_{run_id}"
    assert name in db.table_names(), (
        f"Training table {name} must exist in {LANCEDB_DIR}."
    )
    tbl = db.open_table(name)
    assert tbl.count_rows() == 600, "Training table must have exactly 600 rows."

    schema = tbl.schema
    field_names = [f.name for f in schema]
    for required in ("id", "label", "vector"):
        assert required in field_names, f"Training table missing field '{required}'."

    label_field = schema.field("label")
    assert str(label_field.type) == "int32", (
        f"Field 'label' must be Int32, got {label_field.type}."
    )

    vector_field = schema.field("vector")
    # Expected: fixed_size_list<float32, 40>
    assert vector_field.type.list_size == 40, (
        f"Field 'vector' must be a fixed_size_list of length 40, got {vector_field.type}."
    )
    assert "float" in str(vector_field.type.value_type), (
        f"Field 'vector' must hold floats, got {vector_field.type.value_type}."
    )


def test_training_table_labels_well_formed():
    import lancedb
    from collections import Counter

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(f"train_data_{run_id}")
    df = tbl.to_pandas()
    counts = Counter(int(x) for x in df["label"].tolist())
    assert set(counts.keys()) == {0, 1, 2, 3, 4, 5}, (
        f"Training labels must be exactly {{0..5}}, got {sorted(counts.keys())}."
    )
    for c, n in counts.items():
        assert n == 100, f"Each class must have 100 rows; class {c} has {n}."


def test_test_set_fixture_exists():
    assert os.path.isfile(TEST_SET_PATH), f"Test set fixture {TEST_SET_PATH} must exist."
    with open(TEST_SET_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list), "Test set fixture must be a JSON list."
    assert len(data) == 120, f"Test set fixture must have 120 entries, got {len(data)}."
    for row in data:
        assert isinstance(row, dict), "Each test entry must be a dict."
        assert "vector" in row and "label" in row, "Each entry must have 'vector' and 'label'."
        assert len(row["vector"]) == 40, "Each test vector must have 40 floats."
        assert isinstance(row["label"], int), "Each test label must be an int."
        assert 0 <= row["label"] <= 5, "Test labels must be in {0..5}."


def test_centroids_table_does_not_exist_yet():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(LANCEDB_DIR)
    name = f"centroids_{run_id}"
    assert name not in db.table_names(), (
        f"Candidate must create table {name}; it must not exist in the initial state."
    )
