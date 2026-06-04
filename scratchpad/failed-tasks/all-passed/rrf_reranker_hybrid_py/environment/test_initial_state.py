import importlib
import os

import pytest


PROJECT_DIR = "/home/user/rrf_hybrid"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} must exist before the task starts."
    )


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb must be importable in the task environment."


def test_lancedb_rerankers_importable():
    rerankers = importlib.import_module("lancedb.rerankers")
    assert hasattr(rerankers, "RRFReranker"), (
        "lancedb.rerankers.RRFReranker must be available so the hybrid reranker can be used."
    )


def test_pyarrow_importable():
    pa = importlib.import_module("pyarrow")
    assert pa is not None, "pyarrow must be importable in the task environment."


def test_numpy_importable():
    np = importlib.import_module("numpy")
    assert np is not None, "numpy must be importable in the task environment."


def test_lancedb_uri_default_dir_parent_writable():
    # The task defaults LANCEDB_URI to /workspace/db. The /workspace parent must
    # exist so the candidate's script can create the database directory.
    workspace = "/workspace"
    assert os.path.isdir(workspace), (
        f"Workspace directory {workspace} must exist so LanceDB can be opened at /workspace/db."
    )
    assert os.access(workspace, os.W_OK), (
        f"Workspace directory {workspace} must be writable by the task user."
    )


def test_output_dir_parent_writable():
    workspace = "/workspace"
    # The candidate writes /workspace/output/hybrid_rrf.json; only require that
    # /workspace is writable (the candidate is responsible for mkdir of /workspace/output).
    assert os.access(workspace, os.W_OK), (
        "/workspace must be writable so the candidate can create /workspace/output."
    )
