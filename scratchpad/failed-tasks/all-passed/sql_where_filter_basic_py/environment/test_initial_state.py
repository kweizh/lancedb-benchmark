import importlib
import os

import pytest


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb package is not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow package is not importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy package is not importable."


def test_workspace_directory_exists():
    assert os.path.isdir("/workspace"), "/workspace directory must exist before the task starts."


def test_solution_directory_exists():
    assert os.path.isdir("/workspace/solution"), (
        "/workspace/solution directory must exist as the project path before the task starts."
    )


def test_lancedb_uri_env_var_present():
    # The task description allows defaulting to /workspace/db when LANCEDB_URI is unset,
    # but the environment is expected to provide it.
    assert "LANCEDB_URI" in os.environ, "LANCEDB_URI environment variable must be set."
    assert os.environ["LANCEDB_URI"], "LANCEDB_URI environment variable must not be empty."
