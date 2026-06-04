import importlib
import os

import pytest


PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    module = importlib.import_module("lancedb")
    assert module is not None, "lancedb Python package is not importable."


def test_pyarrow_importable():
    module = importlib.import_module("pyarrow")
    assert module is not None, "pyarrow Python package is not importable."


def test_numpy_importable():
    module = importlib.import_module("numpy")
    assert module is not None, "numpy Python package is not importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist before the task starts."
    )


def test_workspace_db_root_exists():
    assert os.path.isdir("/workspace"), (
        "Workspace root /workspace does not exist before the task starts."
    )


def test_output_dir_exists():
    assert os.path.isdir("/workspace/output"), (
        "Output directory /workspace/output does not exist before the task starts."
    )


def test_distances_json_not_present_initially():
    assert not os.path.exists("/workspace/output/distances.json"), (
        "Pre-existing /workspace/output/distances.json should not be present before the task runs."
    )


def test_lancedb_uri_env_default_or_set():
    # Either LANCEDB_URI is explicitly set or the default location is acceptable.
    # We just assert the variable exists in the environment to ensure the verifier
    # and the candidate can agree on the database location.
    assert "LANCEDB_URI" in os.environ, (
        "LANCEDB_URI environment variable must be defined for the task."
    )
