import os

import pytest


PROJECT_DIR = "/workspace/project"
DEFAULT_LANCEDB_URI = "/workspace/db"
OUTPUT_DIR = "/workspace/output"


def test_lancedb_importable():
    try:
        import lancedb  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic only
        pytest.fail(f"Failed to import lancedb: {exc!r}")


def test_pyarrow_importable():
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic only
        pytest.fail(f"Failed to import pyarrow: {exc!r}")


def test_numpy_importable():
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic only
        pytest.fail(f"Failed to import numpy: {exc!r}")


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Expected project directory at {PROJECT_DIR} to exist before the task starts."
    )


def test_lancedb_uri_parent_writable():
    # The default LANCEDB_URI points at /workspace/db; the parent /workspace
    # must exist and be writable so the executor can create the LanceDB database.
    parent = os.path.dirname(DEFAULT_LANCEDB_URI) or "/"
    assert os.path.isdir(parent), f"Expected {parent} to exist."
    assert os.access(parent, os.W_OK), f"Expected {parent} to be writable."


def test_output_parent_writable():
    parent = os.path.dirname(OUTPUT_DIR) or "/"
    assert os.path.isdir(parent), f"Expected {parent} to exist."
    assert os.access(parent, os.W_OK), f"Expected {parent} to be writable."


def test_counts_json_not_yet_present():
    # The executor must produce this file; it must NOT pre-exist.
    counts_path = os.path.join(OUTPUT_DIR, "counts.json")
    assert not os.path.exists(counts_path), (
        f"Expected {counts_path} to be absent before the task runs, but it exists."
    )
