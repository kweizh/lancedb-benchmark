import os
import importlib

import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_data_dir_exists():
    assert os.path.isdir(DATA_DIR), (
        f"LanceDB data directory {DATA_DIR} does not exist. "
        "The entrypoint script should seed the baseline and current tables before evaluation."
    )


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb package failed to import."


def test_sklearn_importable():
    mod = importlib.import_module("sklearn")
    assert mod is not None, "scikit-learn package failed to import."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow package failed to import."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy package failed to import."


def test_scipy_importable():
    mod = importlib.import_module("scipy")
    assert mod is not None, "scipy package failed to import."


def test_zealt_run_id_env_var_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID env var must be set so the tables can be located."


def test_baseline_table_seeded():
    import lancedb
    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DATA_DIR)
    names = db.table_names()
    expected = f"baseline_{run_id}"
    assert expected in names, (
        f"Expected pre-seeded table {expected!r} under {DATA_DIR}, found tables: {names}"
    )
    tbl = db.open_table(expected)
    assert tbl.count_rows() == 1000, "baseline table must have exactly 1000 rows."


def test_current_table_seeded():
    import lancedb
    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DATA_DIR)
    names = db.table_names()
    expected = f"current_{run_id}"
    assert expected in names, (
        f"Expected pre-seeded table {expected!r} under {DATA_DIR}, found tables: {names}"
    )
    tbl = db.open_table(expected)
    assert tbl.count_rows() == 1000, "current table must have exactly 1000 rows."
