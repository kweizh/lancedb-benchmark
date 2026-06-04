import os

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_uri_env_set():
    uri = os.environ.get("LANCEDB_URI")
    assert uri, "LANCEDB_URI environment variable is not set."
    assert os.path.isdir(uri), f"LanceDB URI directory {uri} does not exist."


def test_table_name_env_set():
    name = os.environ.get("TABLE_NAME")
    assert name, "TABLE_NAME environment variable is not set."


def test_category_filter_env_set():
    val = os.environ.get("CATEGORY_FILTER")
    assert val, "CATEGORY_FILTER environment variable is not set."


def test_fts_query_env_set():
    val = os.environ.get("FTS_QUERY")
    assert val, "FTS_QUERY environment variable is not set."


def test_query_vector_npy_exists():
    qv = os.path.join(PROJECT_DIR, "query_vector.npy")
    assert os.path.isfile(qv), f"Query vector file {qv} does not exist."
    arr = np.load(qv)
    assert arr.ndim == 1 and arr.shape[0] == 16 and arr.dtype == np.float32, (
        f"Query vector must be a 1-D float32 array of length 16, got shape={arr.shape} dtype={arr.dtype}"
    )


def test_table_seeded_with_rows_and_fts_index():
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    name = os.environ["TABLE_NAME"]
    db = lancedb.connect(uri)
    assert name in db.table_names(), f"Table {name} not found in database at {uri}."
    tbl = db.open_table(name)
    n = tbl.count_rows()
    assert n >= 256, f"Table must contain at least 256 rows for IVF/PQ readiness; found {n}."

    indices = tbl.list_indices()
    fts_cols = [list(getattr(i, "columns", [])) for i in indices]
    assert any(cols == ["text"] for cols in fts_cols), (
        f"Expected an FTS index on the 'text' column. Indices found: {indices}"
    )


def test_solution_file_does_not_exist_yet():
    sol = os.path.join(PROJECT_DIR, "solution.py")
    # The executor is expected to create this file; allow either absent or empty.
    if os.path.isfile(sol):
        assert os.path.getsize(sol) == 0, (
            f"{sol} already contains content; the executor should write the solution."
        )
