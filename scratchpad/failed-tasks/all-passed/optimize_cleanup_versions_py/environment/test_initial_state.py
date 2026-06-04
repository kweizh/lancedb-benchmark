import importlib
import os

import pytest


PROJECT_DIR = "/home/user/myproject"


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_workspace_directories_exist():
    # The Dockerfile pre-creates the workspace directories used by the task.
    assert os.path.isdir("/workspace"), "/workspace directory does not exist."
    assert os.path.isdir("/workspace/output"), "/workspace/output directory does not exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb package failed to import."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow package failed to import."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy package failed to import."


def test_lancedb_uri_not_yet_populated():
    # The candidate is responsible for creating the LanceDB store; it must not pre-exist as a populated table.
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    metrics_path = os.path.join(uri, "metrics.lance")
    assert not os.path.exists(metrics_path), (
        f"metrics table directory {metrics_path} must not exist before the task runs."
    )


def test_output_state_file_not_yet_present():
    output_path = "/workspace/output/optimize_state.json"
    assert not os.path.exists(output_path), (
        f"Output file {output_path} must not exist before the task runs."
    )
