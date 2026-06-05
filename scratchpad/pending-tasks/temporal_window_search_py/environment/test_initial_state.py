import os
import shutil

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


def test_python3_available():
    assert shutil.which("python3") is not None, "python3 binary not found in PATH."


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_zealt_run_id_present():
    assert os.environ.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID environment variable must be set in the task environment."
    )


def test_lancedb_data_directory_exists():
    assert os.path.isdir(DB_DIR), (
        f"Pre-seeded LanceDB database directory {DB_DIR} does not exist."
    )


def test_events_table_seeded_with_1000_rows():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"events_{run_id}"
    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert table_name in names, (
        f"Seeded table {table_name!r} not found. Existing tables: {names}"
    )
    table = db.open_table(table_name)
    count = table.count_rows()
    assert count == 1000, (
        f"Seeded events table expected 1000 rows, found {count}."
    )


def test_events_table_schema_has_expected_columns():
    import lancedb
    import pyarrow as pa

    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"events_{run_id}"
    db = lancedb.connect(DB_DIR)
    table = db.open_table(table_name)
    names = set(table.schema.names)
    required = {"id", "timestamp", "event_type", "payload", "vector"}
    missing = required - names
    assert not missing, (
        f"Seeded events table schema is missing required columns: {missing}. "
        f"Found: {sorted(names)}"
    )

    field_types = {f.name: f.type for f in table.schema}
    assert pa.types.is_int64(field_types["id"]), (
        f"`id` column expected Int64, got {field_types['id']}"
    )
    assert pa.types.is_int64(field_types["timestamp"]), (
        f"`timestamp` column expected Int64, got {field_types['timestamp']}"
    )
    vec_type = field_types["vector"]
    assert pa.types.is_fixed_size_list(vec_type) and vec_type.list_size == 32, (
        f"`vector` column expected FixedSizeList<*, 32>, got {vec_type}"
    )
