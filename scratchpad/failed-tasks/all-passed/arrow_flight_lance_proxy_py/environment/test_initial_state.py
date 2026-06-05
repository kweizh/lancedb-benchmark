import os

import pytest

PROJECT_DIR = "/home/user/flight_proxy"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pyarrow_flight_importable():
    import pyarrow.flight  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB directory {LANCEDB_DIR} does not exist."


def test_documents_table_seeded():
    import lancedb

    conn = lancedb.connect(LANCEDB_DIR)
    names = list(conn.table_names())
    assert "documents" in names, f"Expected 'documents' table to exist; found tables: {names}"

    tbl = conn.open_table("documents")
    assert tbl.count_rows() == 200, (
        f"Expected the seeded 'documents' table to have 200 rows; found {tbl.count_rows()}."
    )


def test_documents_schema_has_expected_columns():
    import lancedb
    import pyarrow as pa

    conn = lancedb.connect(LANCEDB_DIR)
    tbl = conn.open_table("documents")
    schema = tbl.schema
    field_names = set(schema.names)
    for col in ("id", "text", "embedding"):
        assert col in field_names, f"Expected column '{col}' in documents schema; got {field_names}."

    emb_type = schema.field("embedding").type
    assert pa.types.is_fixed_size_list(emb_type), (
        f"Expected 'embedding' to be a fixed_size_list type; got {emb_type}."
    )
    assert emb_type.list_size == 32, (
        f"Expected 'embedding' fixed_size_list of width 32; got width {emb_type.list_size}."
    )
