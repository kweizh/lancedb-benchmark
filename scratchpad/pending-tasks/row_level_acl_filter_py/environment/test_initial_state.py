import os

import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "data")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_db_dir_exists():
    assert os.path.isdir(DB_DIR), (
        f"LanceDB directory {DB_DIR} should be pre-seeded by the container entrypoint."
    )


def test_run_id_env_var_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id and run_id.startswith("zr-"), (
        "ZEALT_RUN_ID environment variable must be set to a value starting with 'zr-'."
    )


def test_seeded_documents_table_exists():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"documents_{run_id}"
    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert table_name in names, (
        f"Expected pre-seeded table {table_name!r} in {DB_DIR}, found: {names}"
    )


def test_seeded_table_has_expected_rows_and_schema():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(f"documents_{run_id}")
    assert tbl.count_rows() == 300, (
        f"Pre-seeded documents table should have 300 rows, got {tbl.count_rows()}."
    )

    field_names = {f.name for f in tbl.schema}
    expected = {"id", "text", "owner_id", "visibility", "team_id", "vector"}
    missing = expected - field_names
    assert not missing, f"Pre-seeded table is missing required columns: {missing}"


def test_visibility_values_are_constrained():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(f"documents_{run_id}")
    df = tbl.to_pandas()
    allowed = {"public", "team", "private"}
    actual = set(df["visibility"].unique())
    extras = actual - allowed
    assert not extras, f"visibility column contains unexpected values: {extras}"
    # All three categories must be present so each ACL branch is exercised.
    missing_cats = allowed - actual
    assert not missing_cats, (
        f"Seeded data must contain all visibility classes; missing: {missing_cats}"
    )
