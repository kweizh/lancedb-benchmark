import os

import pytest


PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project dir {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(DATA_DIR), f"LanceDB data dir {DATA_DIR} does not exist."


def test_zealt_run_id_set():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID environment variable must be set."


def test_documents_table_seeded():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DATA_DIR)
    table_name = f"documents_{run_id}"
    assert table_name in db.table_names(), (
        f"Seeded documents table {table_name} not found in {DATA_DIR}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 64, (
        f"Documents table {table_name} should have 64 seed rows."
    )
