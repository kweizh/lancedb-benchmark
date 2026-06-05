import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
GROUND_TRUTH_PATH = os.path.join(PROJECT_DIR, ".ground_truth_outliers.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_sklearn_importable():
    import sklearn  # noqa: F401
    from sklearn.ensemble import IsolationForest  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_zealt_run_id_env_set():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the runtime environment."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB data dir {LANCEDB_DIR} must be pre-seeded."


def test_events_table_exists_with_expected_schema():
    import lancedb
    import pyarrow as pa

    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID env var required to locate the events table."
    table_name = f"events_{run_id}"

    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    assert table_name in names, (
        f"Expected pre-seeded table '{table_name}' in LanceDB; got {names!r}."
    )

    tbl = db.open_table(table_name)
    schema = tbl.schema
    field_names = set(schema.names)
    assert {"id", "ts", "vector"}.issubset(field_names), (
        f"Pre-seeded table missing required columns; have {field_names!r}."
    )

    # Vector must be a fixed-size list of 20 float32 elements.
    vector_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type), (
        f"`vector` column must be a fixed_size_list; got {vector_field.type!r}."
    )
    assert vector_field.type.list_size == 20, (
        f"`vector` column must have list_size=20; got {vector_field.type.list_size}."
    )

    # is_outlier MUST NOT exist yet — it is the column the candidate has to add.
    assert "is_outlier" not in field_names, (
        "`is_outlier` column must not exist in the initial state; the candidate must add it."
    )

    # Row count
    assert tbl.count_rows() == 1000, (
        f"Pre-seeded table must contain exactly 1000 rows; got {tbl.count_rows()}."
    )


def test_ground_truth_outliers_file_present():
    assert os.path.isfile(GROUND_TRUTH_PATH), (
        f"Ground-truth outliers file {GROUND_TRUTH_PATH} must be pre-seeded for the verifier."
    )
    with open(GROUND_TRUTH_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list), "Ground-truth file must contain a JSON list of ints."
    assert len(data) == 50, f"Ground-truth file must contain exactly 50 outlier ids; got {len(data)}."
    assert all(isinstance(x, int) for x in data), "Ground-truth ids must all be integers."
