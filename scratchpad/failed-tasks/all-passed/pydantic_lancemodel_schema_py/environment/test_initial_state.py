import importlib
import os


PROJECT_DIR = "/workspace"
DB_DIR = "/workspace/db"
OUTPUT_DIR = "/workspace/output"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb module is not importable."


def test_lancedb_pydantic_importable():
    mod = importlib.import_module("lancedb.pydantic")
    assert hasattr(mod, "LanceModel"), "lancedb.pydantic.LanceModel is not available."
    assert hasattr(mod, "Vector"), "lancedb.pydantic.Vector is not available."


def test_pydantic_importable():
    pydantic = importlib.import_module("pydantic")
    assert pydantic.VERSION.startswith("2."), (
        f"Expected pydantic v2, got version {pydantic.VERSION}."
    )


def test_numpy_importable():
    np = importlib.import_module("numpy")
    assert np is not None, "numpy is not importable."


def test_pyarrow_importable():
    pa = importlib.import_module("pyarrow")
    assert pa is not None, "pyarrow is not importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_db_dir_exists():
    assert os.path.isdir(DB_DIR), (
        f"LanceDB database directory {DB_DIR} does not exist."
    )


def test_output_dir_exists():
    assert os.path.isdir(OUTPUT_DIR), (
        f"Output directory {OUTPUT_DIR} does not exist."
    )


def test_db_dir_is_empty():
    entries = [e for e in os.listdir(DB_DIR) if not e.startswith(".")]
    assert entries == [], (
        f"Expected {DB_DIR} to be empty before the task runs, found: {entries}."
    )
