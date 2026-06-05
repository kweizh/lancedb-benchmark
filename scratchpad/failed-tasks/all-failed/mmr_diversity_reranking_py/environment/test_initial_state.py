import importlib
import os

import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = "/app/db"


def test_lancedb_importable():
    try:
        lancedb = importlib.import_module("lancedb")
    except Exception as exc:
        pytest.fail(f"lancedb package must be importable in the base environment: {exc}")
    assert hasattr(lancedb, "connect"), "lancedb module is missing the `connect` entrypoint."


def test_pyarrow_importable():
    try:
        pa = importlib.import_module("pyarrow")
    except Exception as exc:
        pytest.fail(f"pyarrow must be importable in the base environment: {exc}")
    assert hasattr(pa, "list_"), "pyarrow.list_ helper is required for the fixed-size vector schema."


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:
        pytest.fail(f"numpy must be importable in the base environment: {exc}")


def test_pandas_importable():
    try:
        importlib.import_module("pandas")
    except Exception as exc:
        pytest.fail(f"pandas must be importable in the base environment: {exc}")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} must exist."


def test_db_parent_dir_exists():
    parent = os.path.dirname(DB_DIR)
    assert os.path.isdir(parent), (
        f"Parent directory {parent} for the LanceDB database must exist so the candidate can write to {DB_DIR}."
    )


def test_zealt_run_id_present():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the task environment."
    assert run_id.startswith("zr-"), f"ZEALT_RUN_ID must start with 'zr-'; got {run_id!r}."
