import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable in the environment."


def test_prometheus_client_importable():
    mod = importlib.import_module("prometheus_client")
    assert mod is not None, "prometheus_client is not importable in the environment."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow is not importable in the environment."


def test_requests_importable():
    mod = importlib.import_module("requests")
    assert mod is not None, "requests is not importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_zealt_run_id_env():
    val = os.environ.get("ZEALT_RUN_ID")
    assert val, "ZEALT_RUN_ID env var must be set for parallel-run isolation."


def test_seeded_lancedb_table_present():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID env var must be set for parallel-run isolation."
    import lancedb

    db_path = os.path.join(PROJECT_DIR, "data", "db")
    assert os.path.isdir(db_path), (
        f"Seeded LanceDB directory {db_path} does not exist; entrypoint seeding may have failed."
    )
    db = lancedb.connect(db_path)
    table_name = f"documents_{run_id}"
    assert table_name in db.table_names(), (
        f"Expected seeded table {table_name} to exist in {db_path}; got {db.table_names()}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 200, (
        f"Expected seeded table {table_name} to have 200 rows, got {tbl.count_rows()}."
    )
