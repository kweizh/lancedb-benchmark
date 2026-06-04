import os

import pytest


PROJECT_DIR = "/home/user/myproject"
LANCEDB_PATH = "/data/lancedb"
EXPECTED_PATH = "/opt/expected_diffs.json"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_database_dir_exists():
    assert os.path.isdir(LANCEDB_PATH), f"LanceDB database directory {LANCEDB_PATH} does not exist."


def test_customers_table_present_and_has_versions():
    import lancedb

    db = lancedb.connect(LANCEDB_PATH)
    names = db.table_names()
    assert "customers" in names, f"LanceDB table 'customers' not found at {LANCEDB_PATH}; saw {names!r}."

    table = db.open_table("customers")
    versions = table.list_versions()
    assert len(versions) >= 4, (
        f"Expected at least 4 versions of the 'customers' table, got {len(versions)}: {versions!r}."
    )


def test_customers_schema_has_expected_columns():
    import lancedb

    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table("customers")
    schema = table.schema
    field_names = set(schema.names)
    for expected in ("id", "name", "price", "category"):
        assert expected in field_names, (
            f"Expected column '{expected}' in 'customers' schema, got {sorted(field_names)!r}."
        )


def test_expected_diffs_fixture_present():
    assert os.path.isfile(EXPECTED_PATH), (
        f"Expected diff fixture {EXPECTED_PATH} not found; seed script did not run correctly."
    )
    import json

    with open(EXPECTED_PATH) as f:
        data = json.load(f)
    for key in ("1_4", "2_3", "3_4"):
        assert key in data, f"Expected fixture key '{key}' missing from {EXPECTED_PATH}; got {sorted(data.keys())!r}."
        for sub in ("added", "removed", "modified"):
            assert sub in data[key], (
                f"Expected fixture[{key!r}] missing '{sub}' key; got {sorted(data[key].keys())!r}."
            )
