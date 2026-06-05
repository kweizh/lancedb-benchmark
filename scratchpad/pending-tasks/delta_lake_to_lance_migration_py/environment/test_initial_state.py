import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DELTA_PATH = "/app/delta_data/products"


def test_python3_available():
    import shutil

    assert shutil.which("python3") is not None, "python3 binary not found in PATH."


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_deltalake_importable():
    import deltalake  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} is missing."


def test_delta_table_dir_exists():
    assert os.path.isdir(DELTA_PATH), f"Seeded Delta table directory {DELTA_PATH} is missing."


def test_delta_log_exists():
    log_dir = os.path.join(DELTA_PATH, "_delta_log")
    assert os.path.isdir(log_dir), f"_delta_log directory missing under {DELTA_PATH}."


def test_delta_table_has_three_versions():
    from deltalake import DeltaTable

    dt = DeltaTable(DELTA_PATH)
    # Latest version index should be 2 (0,1,2 => 3 commits).
    assert dt.version() == 2, f"Expected latest Delta version 2, got {dt.version()}."


def test_delta_latest_row_count_is_800():
    from deltalake import DeltaTable

    dt = DeltaTable(DELTA_PATH)
    tbl = dt.to_pyarrow_table()
    assert tbl.num_rows == 800, f"Expected 800 rows in latest Delta version, got {tbl.num_rows}."


def test_zealt_run_id_env_present():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID env var must be set in the environment."
