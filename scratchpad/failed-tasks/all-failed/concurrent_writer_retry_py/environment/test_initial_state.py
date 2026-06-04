import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb module is not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow module is not importable."


def test_lancedb_uri_env_is_set():
    assert os.environ.get("LANCEDB_URI"), "LANCEDB_URI environment variable is not set."


def test_lancedb_table_env_is_set():
    assert os.environ.get("LANCEDB_TABLE"), "LANCEDB_TABLE environment variable is not set."


def test_target_table_seeded_with_200_rows():
    """The Docker image seeds the target table with 200 rows (id=0..199, value=0, ts=0)."""
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    name = os.environ["LANCEDB_TABLE"]
    db = lancedb.connect(uri)
    assert name in db.table_names(), (
        f"Target table {name!r} is missing from LanceDB at {uri!r}."
    )
    tbl = db.open_table(name)
    assert tbl.count_rows() == 200, (
        f"Target table {name!r} should be pre-seeded with exactly 200 rows; "
        f"got {tbl.count_rows()}."
    )


def test_write_attempts_table_is_absent():
    """The write_attempts table must NOT exist at the start of the task; the candidate creates it."""
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    db = lancedb.connect(uri)
    assert "write_attempts" not in db.table_names(), (
        "write_attempts table must not exist at the start; the SafeWriter is "
        "responsible for creating it on first use."
    )
