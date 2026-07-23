import os

import pytest

PROJECT_DIR = "/home/user/myproject"
EXPECTED_ROWS = 5000
EXPECTED_DIM = 64


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_env_vars_present():
    assert os.environ.get("LANCEDB_URI"), "LANCEDB_URI environment variable is not set."
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID environment variable is not set."


def test_lancedb_uri_dir_exists():
    uri = os.environ.get("LANCEDB_URI", "/app/lancedb")
    assert os.path.isdir(uri), f"LanceDB directory {uri} does not exist."


def _open_table():
    import lancedb

    uri = os.environ.get("LANCEDB_URI", "/app/lancedb")
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    table_name = f"vectors_{run_id}"
    db = lancedb.connect(uri)
    names = db.table_names()
    assert table_name in names, (
        f"Seeded table {table_name!r} not found in LanceDB. Available: {list(names)}"
    )
    return db.open_table(table_name)


def test_seeded_table_row_count():
    tbl = _open_table()
    assert tbl.count_rows() == EXPECTED_ROWS, (
        f"Seeded table should contain {EXPECTED_ROWS} rows, found {tbl.count_rows()}."
    )


def test_seeded_table_schema():
    import pyarrow as pa

    tbl = _open_table()
    schema = tbl.schema
    names = set(schema.names)
    assert "id" in names, "Seeded table is missing the required integer 'id' column."
    assert "vector" in names, "Seeded table is missing the required 'vector' column."

    vec_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vec_field.type), (
        "'vector' column must be a fixed-size list type."
    )
    assert vec_field.type.list_size == EXPECTED_DIM, (
        f"'vector' column must have dimension {EXPECTED_DIM}, "
        f"found {vec_field.type.list_size}."
    )


def test_no_vector_index_yet():
    # The candidate is expected to build the ANN index; it must not exist initially.
    tbl = _open_table()
    indices = tbl.list_indices()
    assert len(indices) == 0, (
        "No vector index should exist in the initial state; "
        f"found: {indices}"
    )
