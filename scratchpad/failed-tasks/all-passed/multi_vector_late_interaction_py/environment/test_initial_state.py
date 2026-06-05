import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
FIXTURE_PATH = os.path.join(PROJECT_DIR, ".fixture.npz")
TABLE_NAME = "colbert_tokens"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} is missing."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data directory {LANCEDB_DIR} is missing."


def test_fixture_npz_exists():
    assert os.path.isfile(FIXTURE_PATH), f"Fixture file {FIXTURE_PATH} is missing."


def test_seeded_colbert_table_present():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    assert TABLE_NAME in names, (
        f"Expected seeded table '{TABLE_NAME}' in {LANCEDB_DIR}; actual tables: {names}."
    )
    tbl = db.open_table(TABLE_NAME)
    rows = tbl.count_rows()
    assert rows == 240, f"Expected 240 rows (60 docs x 4 tokens) but got {rows}."


def test_table_schema_columns():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    schema = tbl.schema
    field_names = [f.name for f in schema]
    for required in ("doc_id", "token_idx", "embedding"):
        assert required in field_names, f"Missing column {required} in table schema {field_names}."


def test_fixture_npz_has_expected_keys():
    import numpy as np

    with np.load(FIXTURE_PATH) as data:
        keys = set(data.files)
    required = {"Q1", "Q2", "rigged_doc1", "rigged_doc2", "decoy_doc_id"}
    missing = required - keys
    assert not missing, f"Fixture .fixture.npz is missing keys: {missing}; found: {keys}."
