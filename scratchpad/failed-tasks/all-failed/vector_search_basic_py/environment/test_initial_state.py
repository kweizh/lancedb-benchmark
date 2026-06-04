import importlib
import os

import pytest


PROJECT_DIR = "/workspace/myproject"


def test_lancedb_importable():
    """The target library lancedb must be installed in the runtime environment."""
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"lancedb is not importable in the runtime environment: {exc!r}")


def test_pyarrow_importable():
    """PyArrow is required for declaring the fixed-size-list vector schema."""
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"pyarrow is not importable in the runtime environment: {exc!r}")


def test_numpy_importable():
    """numpy is required for the deterministic seeded RNG used to build vectors."""
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"numpy is not importable in the runtime environment: {exc!r}")


def test_project_directory_exists():
    """The task explicitly declares /workspace/myproject as the project path."""
    assert os.path.isdir(PROJECT_DIR), (
        f"Expected project directory {PROJECT_DIR} to be pre-created in the environment."
    )
