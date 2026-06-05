import os

import pytest


PROJECT_DIR = "/home/user/myproject"


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_lancedb_async_connect_available():
    import lancedb

    assert hasattr(lancedb, "connect_async"), (
        "lancedb.connect_async is required for this task but is missing."
    )


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_zealt_run_id_env_var_present():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set for parallel-safe table naming."


def test_no_prebuilt_solution():
    # The executor is expected to create solution.py and run.py.
    # We only verify that the directory is empty of those artifacts at start.
    for fname in ("solution.py", "run.py", "flush_log.txt"):
        path = os.path.join(PROJECT_DIR, fname)
        assert not os.path.exists(path), (
            f"{path} must not exist before the task starts; the executor is expected to create it."
        )
