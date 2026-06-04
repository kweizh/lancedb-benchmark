import json
import os

import pytest

PROJECT_DIR = "/home/user/project"
DATA_DIR = "/home/user/project/data"
QUERY_PATH = "/home/user/project/query.npy"
EXPECTED_PATH = "/opt/zealt/expected.json"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(DATA_DIR), f"LanceDB data directory {DATA_DIR} does not exist."


def test_query_vector_file_exists():
    assert os.path.isfile(QUERY_PATH), f"Query vector file {QUERY_PATH} does not exist."


def test_expected_file_exists():
    assert os.path.isfile(EXPECTED_PATH), f"Expected results file {EXPECTED_PATH} does not exist."


def test_items_table_present():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    names = set(db.table_names())
    assert "items" in names, f"items table missing in {DATA_DIR}; found: {names}"


def test_user_history_table_present():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    names = set(db.table_names())
    assert "user_history" in names, f"user_history table missing in {DATA_DIR}; found: {names}"


def test_items_schema_has_vector_column():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table("items")
    fields = {f.name for f in tbl.schema}
    assert {"id", "title", "category", "vector"}.issubset(fields), (
        f"items table missing expected columns; got: {fields}"
    )


def test_user_history_has_target_user():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table("user_history")
    df = tbl.search().where("user_id = 'u_test'").limit(100).to_pandas()
    assert len(df) >= 10, (
        f"Target user u_test must have at least 10 history rows, got {len(df)}"
    )


def test_expected_json_is_well_formed():
    with open(EXPECTED_PATH) as f:
        data = json.load(f)
    for key in ("expected_alpha0", "expected_alpha1", "seen_item_ids"):
        assert key in data, f"Missing key {key!r} in {EXPECTED_PATH}"
    assert len(data["expected_alpha0"]) == 5
    assert len(data["expected_alpha1"]) == 5
    assert data["expected_alpha0"] != data["expected_alpha1"], (
        "Fixture is malformed: alpha=0 and alpha=1 expected rankings must differ."
    )


def test_recommend_script_not_present():
    """The candidate is expected to CREATE this file."""
    assert not os.path.exists(os.path.join(PROJECT_DIR, "recommend.py")), (
        "recommend.py should not exist before the candidate runs."
    )
