import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project dir {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data dir {LANCEDB_DIR} does not exist."


def test_zealt_run_id_env_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the environment."


def test_seed_table_present_with_50_rows():
    import lancedb

    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID is required"
    db = lancedb.connect(LANCEDB_DIR)
    table_name = f"phrase_docs_{run_id}"
    assert table_name in db.table_names(), (
        f"Pre-seeded table {table_name} not found in {LANCEDB_DIR}; tables: {db.table_names()}"
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 50, (
        f"Pre-seeded table {table_name} expected to have 50 rows, got {tbl.count_rows()}"
    )


def test_ground_truth_fixture_file_exists():
    # The seed step also persists the verifier-facing ground-truth id sets.
    gt_path = "/app/ground_truth.json"
    assert os.path.isfile(gt_path), (
        f"Ground-truth fixture {gt_path} (written by the seed step) is missing."
    )
