import importlib
import os


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not importable in the environment."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow is not importable in the environment."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy is not importable in the environment."


def test_workspace_dir_exists():
    assert os.path.isdir("/workspace"), "/workspace project directory is missing."


def test_output_dir_exists():
    assert os.path.isdir("/workspace/output"), (
        "/workspace/output directory is missing; the task writes its result file here."
    )


def test_lancedb_uri_env_default_present():
    # The Dockerfile must declare LANCEDB_URI so the candidate can read it.
    assert os.environ.get("LANCEDB_URI"), (
        "LANCEDB_URI environment variable is not set in the task environment."
    )
