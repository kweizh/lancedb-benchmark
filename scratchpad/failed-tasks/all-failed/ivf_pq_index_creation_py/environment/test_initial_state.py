import os
import importlib

PROJECT_DIR = "/home/user/myproject"
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_DIR = "/workspace/output"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Expected project directory {PROJECT_DIR} to exist before the task starts."
    )


def test_lancedb_uri_parent_dir_exists():
    parent = os.path.dirname(LANCEDB_URI.rstrip("/")) or "/"
    assert os.path.isdir(parent), (
        f"Expected parent directory {parent} of LANCEDB_URI ({LANCEDB_URI}) to exist."
    )


def test_output_dir_exists():
    assert os.path.isdir(OUTPUT_DIR), (
        f"Expected output directory {OUTPUT_DIR} to exist before the task starts."
    )


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb module is not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow module is not importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy module is not importable."


def test_output_artifact_not_yet_present():
    # The candidate is expected to create this file; it must not pre-exist.
    artifact = os.path.join(OUTPUT_DIR, "ivf_pq.json")
    assert not os.path.exists(artifact), (
        f"Expected {artifact} to NOT exist before the task starts."
    )
