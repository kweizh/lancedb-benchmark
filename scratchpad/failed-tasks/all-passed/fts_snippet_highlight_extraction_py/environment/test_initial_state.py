import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb Python SDK is not importable in the environment."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow is not importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lance_db_directory_exists():
    assert os.path.isdir(DATA_DIR), f"LanceDB database directory {DATA_DIR} does not exist."


def test_lance_table_env_var_set():
    name = os.environ.get("LANCE_TABLE")
    assert name, "LANCE_TABLE environment variable must be set to the fixture table name."


def test_fixture_table_open_and_has_50_rows():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    table_name = os.environ["LANCE_TABLE"]
    assert table_name in db.table_names(), (
        f"Fixture table {table_name!r} not found in LanceDB at {DATA_DIR}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 50, (
        f"Fixture table {table_name!r} must contain exactly 50 rigged article rows."
    )


def test_fts_index_present_on_body():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table(os.environ["LANCE_TABLE"])
    indices = tbl.list_indices()
    # At least one FTS index covering the body column must exist.
    has_body_fts = any(
        getattr(idx, "columns", None) == ["body"]
        and getattr(idx, "index_type", "").upper().startswith("FTS")
        for idx in indices
    )
    assert has_body_fts, (
        "Expected an FTS index on the 'body' column of the fixture table."
    )
