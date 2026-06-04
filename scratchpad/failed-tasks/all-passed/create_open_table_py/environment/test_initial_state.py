import importlib
import os

import pytest


PROJECT_DIR = "/home/user/myproject"
WORKSPACE_DB = "/workspace/db"
WORKSPACE_OUTPUT = "/workspace/output"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_workspace_db_parent_exists():
    # /workspace must exist so that the candidate can create /workspace/db.
    assert os.path.isdir("/workspace"), "/workspace does not exist."


def test_workspace_output_dir_exists():
    assert os.path.isdir(WORKSPACE_OUTPUT), f"Workspace output directory {WORKSPACE_OUTPUT} does not exist."


def test_workspace_db_not_initialized():
    # The candidate is expected to create the LanceDB database during the task.
    assert not os.path.isdir(WORKSPACE_DB), (
        f"LanceDB database directory {WORKSPACE_DB} should not exist before the task runs."
    )


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"lancedb is not importable in the task environment: {exc!r}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"pyarrow is not importable in the task environment: {exc!r}")


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"numpy is not importable in the task environment: {exc!r}")
