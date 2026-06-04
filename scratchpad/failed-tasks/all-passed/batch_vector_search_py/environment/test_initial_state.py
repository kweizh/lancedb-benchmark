import importlib
import os


PROJECT_DIR = "/home/user/myproject"
DB_DIR = "/workspace/db"
OUTPUT_DIR = "/workspace/output"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb Python package is not importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy Python package is not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow Python package is not importable."


def test_pandas_importable():
    mod = importlib.import_module("pandas")
    assert mod is not None, "pandas Python package is not importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_db_parent_dir_exists():
    # /workspace must exist so the candidate can create /workspace/db and /workspace/output.
    assert os.path.isdir("/workspace"), "/workspace directory does not exist."


def test_db_dir_is_empty_or_absent():
    # The candidate is responsible for creating the LanceDB table; we should not have it pre-seeded.
    if os.path.isdir(DB_DIR):
        assert not os.listdir(DB_DIR), f"{DB_DIR} should be empty before the task runs."


def test_output_file_absent():
    output_file = os.path.join(OUTPUT_DIR, "batch_search.json")
    assert not os.path.exists(output_file), (
        f"{output_file} should not exist before the task runs; it must be produced by the candidate."
    )
