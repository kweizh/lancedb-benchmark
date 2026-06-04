import os
import shutil


PROJECT_DIR = "/home/user/myproject"
DEFAULT_LANCEDB_URI = "/workspace/db"
OUTPUT_DIR = "/workspace/output"


def test_python3_available():
    assert shutil.which("python3") is not None, "python3 binary not found in PATH."


def test_lancedb_importable():
    import lancedb  # noqa: F401

    assert hasattr(lancedb, "connect"), "lancedb.connect is not available; the SDK is not installed properly."


def test_pyarrow_importable():
    import pyarrow as pa  # noqa: F401

    assert hasattr(pa, "table"), "pyarrow.table is not available; pyarrow is not installed properly."


def test_numpy_importable():
    import numpy as np  # noqa: F401

    assert hasattr(np, "random"), "numpy.random is not available; numpy is not installed properly."


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_workspace_directory_exists():
    # The candidate's script writes to /workspace/db and /workspace/output; the parent must already be present.
    assert os.path.isdir("/workspace"), "Workspace directory /workspace does not exist."


def test_lancedb_uri_not_yet_populated():
    # The candidate is responsible for creating the LanceDB database. If anything already lives at the
    # default URI it must NOT contain the `users` table created by the task.
    if os.path.exists(DEFAULT_LANCEDB_URI):
        users_dir = os.path.join(DEFAULT_LANCEDB_URI, "users.lance")
        assert not os.path.exists(users_dir), (
            f"Pre-existing LanceDB table {users_dir} should not be present before the task runs."
        )


def test_output_file_not_present_yet():
    output_file = os.path.join(OUTPUT_DIR, "upsert_state.json")
    assert not os.path.exists(output_file), (
        f"Output artifact {output_file} should not exist before the candidate's script runs."
    )
