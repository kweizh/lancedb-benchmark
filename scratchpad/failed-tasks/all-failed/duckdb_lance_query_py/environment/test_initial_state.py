import importlib
import os

PROJECT_DIR = "/home/user/myproject"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable."


def test_lance_importable():
    mod = importlib.import_module("lance")
    assert mod is not None, "pylance (lance) is not importable."


def test_duckdb_importable():
    mod = importlib.import_module("duckdb")
    assert mod is not None, "duckdb is not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow is not importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy is not importable."


def test_zealt_run_id_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the task environment."
