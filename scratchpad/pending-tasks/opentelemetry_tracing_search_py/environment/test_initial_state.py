import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable in the environment."


def test_opentelemetry_api_importable():
    mod = importlib.import_module("opentelemetry")
    assert mod is not None, "opentelemetry api package is not importable."


def test_opentelemetry_sdk_importable():
    mod = importlib.import_module("opentelemetry.sdk.trace")
    assert mod is not None, "opentelemetry.sdk.trace is not importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} is missing."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data directory {LANCEDB_DIR} is missing."


def test_zealt_run_id_env_var_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set before evaluation."


def test_seeded_table_present():
    import lancedb
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID is required to determine the seeded table name."
    db = lancedb.connect(LANCEDB_DIR)
    expected = f"tracing_docs_{run_id}"
    names = list(db.table_names())
    assert expected in names, (
        f"Expected seeded table '{expected}' to be present in LanceDB; "
        f"found tables: {names}"
    )
    tbl = db.open_table(expected)
    assert tbl.count_rows() == 50, (
        f"Seeded table {expected} should contain exactly 50 rows; "
        f"found {tbl.count_rows()}."
    )
