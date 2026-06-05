import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Expected project directory {PROJECT_DIR} to exist before the task starts."
    )


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - import surface
        pytest.fail(f"lancedb must be importable in the initial environment: {exc!r}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - import surface
        pytest.fail(f"pyarrow must be importable in the initial environment: {exc!r}")


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - import surface
        pytest.fail(f"numpy must be importable in the initial environment: {exc!r}")


def test_lance_db_path_env_var_present():
    assert os.environ.get("LANCE_DB_PATH"), (
        "LANCE_DB_PATH environment variable must be set in the initial environment."
    )


def test_zealt_run_id_env_var_present():
    assert os.environ.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID environment variable must be set in the initial environment."
    )
