import os
import importlib


PROJECT_DIR = "/home/user/myproject"


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} must exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb must be importable."


def test_polars_importable():
    mod = importlib.import_module("polars")
    assert mod is not None, "polars must be importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow must be importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy must be importable."


def test_zealt_run_id_env_present():
    assert os.environ.get("ZEALT_RUN_ID"), \
        "ZEALT_RUN_ID environment variable must be set in the task environment."
