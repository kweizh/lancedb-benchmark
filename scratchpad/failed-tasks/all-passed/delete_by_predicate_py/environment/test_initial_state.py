import importlib
import os

import pytest


PROJECT_DIR = "/home/user/lance_delete"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb Python package must be importable in the environment."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow Python package must be importable in the environment."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy Python package must be importable in the environment."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Expected project directory {PROJECT_DIR} to exist before the task begins."
    )


def test_workspace_db_root_exists():
    assert os.path.isdir("/workspace"), (
        "Expected /workspace directory to exist as the LanceDB working root."
    )


def test_output_dir_exists():
    assert os.path.isdir("/workspace/output"), (
        "Expected /workspace/output directory to exist for the candidate's JSON artifact."
    )
