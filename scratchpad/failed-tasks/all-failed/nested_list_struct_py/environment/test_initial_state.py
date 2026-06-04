import importlib
import os


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb module is not importable in the initial environment."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow module is not importable in the initial environment."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy module is not importable in the initial environment."


def test_solution_dir_exists():
    assert os.path.isdir("/workspace/solution"), (
        "Expected /workspace/solution directory to exist in the initial environment."
    )


def test_output_dir_exists():
    assert os.path.isdir("/workspace/output"), (
        "Expected /workspace/output directory to exist in the initial environment."
    )


def test_db_dir_exists():
    assert os.path.isdir("/workspace/db"), (
        "Expected /workspace/db directory to exist in the initial environment."
    )
