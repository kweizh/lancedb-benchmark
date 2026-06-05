import json
import os

import pytest


PROJECT_DIR = "/home/user/myproject"
DB_PATH = "/home/user/myproject/lancedb_data"
POINTER_PATH = "/app/index_pointer.json"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_zealt_run_id_env_present():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID env var must be set in the container."


def _base_table_name():
    return f"vectors_{os.environ['ZEALT_RUN_ID']}"


def test_lancedb_db_path_exists():
    assert os.path.isdir(DB_PATH), f"LanceDB data dir {DB_PATH} does not exist."


def test_pointer_file_initialized():
    assert os.path.isfile(POINTER_PATH), f"Pointer file {POINTER_PATH} does not exist."
    with open(POINTER_PATH) as f:
        data = json.load(f)
    assert "active" in data, f"Pointer file must contain an 'active' key: {data!r}"
    expected = _base_table_name()
    assert data["active"] == expected, (
        f"Pointer file should start pointing at base table {expected!r}, got {data['active']!r}"
    )


def test_base_table_seeded_with_index():
    import lancedb

    db = lancedb.connect(DB_PATH)
    base = _base_table_name()
    assert base in db.table_names(), f"Base table {base} not found in LanceDB. Got {db.table_names()}"
    tbl = db.open_table(base)
    assert tbl.count_rows() >= 300, (
        f"Base table must be seeded with at least 300 rows for IVF_PQ training, got {tbl.count_rows()}"
    )
    indices = tbl.list_indices()
    vector_indices = [i for i in indices if list(i.columns) == ["vector"]]
    assert len(vector_indices) >= 1, (
        f"Base table must have at least one vector index pre-built; got list_indices()={indices!r}"
    )
