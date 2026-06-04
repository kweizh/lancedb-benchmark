import importlib
import os

import pytest


WORKSPACE = "/workspace"
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - test environment guard
        pytest.fail(f"lancedb is not importable in the task environment: {exc!r}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - test environment guard
        pytest.fail(f"pyarrow is not importable in the task environment: {exc!r}")


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - test environment guard
        pytest.fail(f"numpy is not importable in the task environment: {exc!r}")


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Workspace directory {WORKSPACE} does not exist before the task starts."
    )


def test_lancedb_uri_parent_writable():
    parent = os.path.dirname(LANCEDB_URI.rstrip("/")) or "/"
    assert os.path.isdir(parent), (
        f"Parent directory {parent} for LANCEDB_URI={LANCEDB_URI} does not exist."
    )
    assert os.access(parent, os.W_OK), (
        f"Parent directory {parent} for LANCEDB_URI={LANCEDB_URI} is not writable."
    )


def test_output_directory_writable():
    output_parent = os.path.join(WORKSPACE, "output")
    # The output directory may or may not exist yet; the workspace must at least be writable.
    assert os.access(WORKSPACE, os.W_OK), (
        f"Workspace {WORKSPACE} is not writable; cannot create {output_parent}."
    )


def test_notes_table_not_yet_created():
    # The candidate is responsible for creating the `notes` table. It must NOT exist beforehand.
    table_dir = os.path.join(LANCEDB_URI, "notes.lance")
    assert not os.path.exists(table_dir), (
        f"LanceDB table directory {table_dir} should not exist before the task runs."
    )
