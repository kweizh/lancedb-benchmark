import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
QUERY_PATH = os.path.join(PROJECT_DIR, "query.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(DB_DIR), f"LanceDB data directory {DB_DIR} does not exist."


def test_query_json_exists():
    assert os.path.isfile(QUERY_PATH), f"Query fixture {QUERY_PATH} does not exist."


def test_query_json_shape():
    with open(QUERY_PATH) as f:
        data = json.load(f)
    assert "q0" in data and "relevant_ids" in data, (
        "query.json must contain 'q0' and 'relevant_ids' keys."
    )
    q0 = data["q0"]
    rel = data["relevant_ids"]
    assert isinstance(q0, list) and len(q0) == 32, "q0 must be a 32-length list of floats."
    assert all(isinstance(x, (int, float)) for x in q0), "q0 must contain only numbers."
    assert rel == [50, 51, 52, 53, 54], (
        "relevant_ids in the fixture must be [50, 51, 52, 53, 54]."
    )


def test_zealt_run_id_set():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the task environment."


def test_seeded_table_exists_with_400_rows():
    import lancedb

    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID must be set."
    db = lancedb.connect(DB_DIR)
    table_name = f"documents_{run_id}"
    names = db.table_names()
    assert table_name in names, (
        f"Pre-seeded table '{table_name}' not found in {DB_DIR}. Available: {names}"
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 400, (
        f"Pre-seeded table must contain exactly 400 rows, found {tbl.count_rows()}."
    )
