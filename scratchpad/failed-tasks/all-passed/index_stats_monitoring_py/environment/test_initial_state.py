import os

import pytest

PROJECT_DIR = "/home/user/project"
WORKSPACE_DB = "/workspace/db"
WORKSPACE_OUTPUT = "/workspace/output"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_workspace_db_dir_exists():
    assert os.path.isdir(WORKSPACE_DB), (
        f"LanceDB workspace directory {WORKSPACE_DB} does not exist."
    )


def test_workspace_output_dir_exists():
    assert os.path.isdir(WORKSPACE_OUTPUT), (
        f"Workspace output directory {WORKSPACE_OUTPUT} does not exist."
    )


def test_no_preexisting_artifact():
    artifact = os.path.join(WORKSPACE_OUTPUT, "index_stats.json")
    assert not os.path.exists(artifact), (
        f"Unexpected pre-existing artifact at {artifact}; executor must create it."
    )


def test_lancedb_version_pinned():
    import lancedb

    assert lancedb.__version__.startswith("0.25."), (
        f"Expected lancedb 0.25.x to be installed, found {lancedb.__version__}."
    )


def test_lancedb_uri_env_default(monkeypatch):
    monkeypatch.delenv("LANCEDB_URI", raising=False)
    assert os.environ.get("LANCEDB_URI") is None
