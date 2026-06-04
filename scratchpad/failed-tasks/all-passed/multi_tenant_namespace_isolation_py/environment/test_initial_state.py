import os
import importlib


PROJECT_DIR = "/home/user/myproject"


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} must exist before the task starts."
    )


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert hasattr(mod, "connect"), (
        "lancedb is installed but the top-level `connect` API is missing."
    )


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert hasattr(mod, "schema"), (
        "pyarrow is installed but `pyarrow.schema` is missing."
    )


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert hasattr(mod, "random"), (
        "numpy is installed but `numpy.random` is missing."
    )


def test_pandas_importable():
    mod = importlib.import_module("pandas")
    assert hasattr(mod, "DataFrame"), (
        "pandas is installed but `pandas.DataFrame` is missing."
    )
