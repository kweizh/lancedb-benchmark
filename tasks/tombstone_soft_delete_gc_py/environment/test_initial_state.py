import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_project_directory_exists():
    assert os.path.isdir(
        PROJECT_DIR
    ), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import lancedb: {exc}")


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import numpy: {exc}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import pyarrow: {exc}")


def test_lance_importable():
    # The verifier inspects physical fragments via table.to_lance().get_fragments(),
    # which requires the `lance` (pylance) bindings to be installed.
    try:
        importlib.import_module("lance")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import lance (pylance): {exc}")
