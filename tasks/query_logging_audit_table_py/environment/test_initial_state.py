import os

import lancedb
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
DB_PATH = os.path.join(PROJECT_DIR, "data.lancedb")
ARTICLES_TABLE = "articles"


def test_lancedb_importable():
    import lancedb  # noqa: F401

    assert lancedb is not None, "Failed to import the lancedb Python SDK."


def test_pyarrow_importable():
    import pyarrow  # noqa: F401

    assert pyarrow is not None, "Failed to import pyarrow."


def test_numpy_importable():
    import numpy  # noqa: F401

    assert numpy is not None, "Failed to import numpy."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_db_path_exists():
    assert os.path.isdir(DB_PATH), f"Expected LanceDB database directory {DB_PATH} to exist."


def test_articles_table_exists_and_has_200_rows():
    db = lancedb.connect(DB_PATH)
    names = db.table_names()
    assert ARTICLES_TABLE in names, f"Expected table '{ARTICLES_TABLE}' inside {DB_PATH}, found {names}."
    tbl = db.open_table(ARTICLES_TABLE)
    assert tbl.count_rows() == 200, f"Expected articles table to have 200 rows, got {tbl.count_rows()}."


def test_articles_schema_has_required_columns():
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(ARTICLES_TABLE)
    schema = tbl.schema
    field_names = {f.name for f in schema}
    for col in ("id", "title", "embedding"):
        assert col in field_names, f"Articles table is missing required column '{col}'. Got: {field_names}"

    embed_field = schema.field("embedding")
    assert pa.types.is_fixed_size_list(embed_field.type), (
        f"Expected 'embedding' column to be a fixed_size_list, got {embed_field.type}."
    )
    assert embed_field.type.list_size == 64, (
        f"Expected 'embedding' column to have list size 64, got {embed_field.type.list_size}."
    )


def test_query_logs_table_does_not_exist_yet():
    db = lancedb.connect(DB_PATH)
    names = db.table_names()
    assert "query_logs" not in names, (
        "Expected 'query_logs' table to NOT exist before the candidate runs; the candidate must create it lazily."
    )
