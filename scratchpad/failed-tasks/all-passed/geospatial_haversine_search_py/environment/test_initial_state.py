import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
QUERY_VEC_A = os.path.join(PROJECT_DIR, "query_vec_A.json")
QUERY_VEC_B = os.path.join(PROJECT_DIR, "query_vec_B.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(DB_DIR), f"LanceDB directory {DB_DIR} does not exist."


def test_pois_table_exists_and_has_rows():
    import lancedb

    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert "pois" in names, f"Expected table 'pois' in {names}."
    table = db.open_table("pois")
    n = table.count_rows()
    assert n >= 200, f"Expected at least 200 rows in 'pois' table, found {n}."


def test_pois_table_schema():
    import lancedb

    db = lancedb.connect(DB_DIR)
    table = db.open_table("pois")
    field_names = {f.name for f in table.schema}
    expected = {"id", "name", "lat", "lon", "category", "embedding"}
    missing = expected - field_names
    assert not missing, f"Missing expected columns: {missing} (found {field_names})."


def test_query_vector_a_exists():
    assert os.path.isfile(QUERY_VEC_A), f"{QUERY_VEC_A} does not exist."
    with open(QUERY_VEC_A) as f:
        v = json.load(f)
    assert isinstance(v, list) and len(v) == 32, "Query vector A must be a 32-element list."


def test_query_vector_b_exists():
    assert os.path.isfile(QUERY_VEC_B), f"{QUERY_VEC_B} does not exist."
    with open(QUERY_VEC_B) as f:
        v = json.load(f)
    assert isinstance(v, list) and len(v) == 32, "Query vector B must be a 32-element list."
