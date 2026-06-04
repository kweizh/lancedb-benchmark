import importlib
import os

import pytest


WORKSPACE = "/workspace"
SOLUTION_DIR = "/workspace/solution"
OUTPUT_DIR = "/workspace/output"
DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "inventory"
EXPECTED_ROW_COUNT = 12


def test_lancedb_importable():
    """The lancedb Python SDK must be installed and importable."""
    lancedb = importlib.import_module("lancedb")
    assert lancedb is not None, "Expected the lancedb Python SDK to be importable."


def test_pyarrow_importable():
    """pyarrow is required for inspecting the lancedb table schema."""
    pa = importlib.import_module("pyarrow")
    assert pa is not None, "Expected pyarrow to be importable."


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Expected workspace directory {WORKSPACE} to exist."


def test_solution_directory_exists():
    assert os.path.isdir(SOLUTION_DIR), (
        f"Expected the candidate solution directory {SOLUTION_DIR} to exist."
    )


def test_output_directory_exists():
    assert os.path.isdir(OUTPUT_DIR), (
        f"Expected the output directory {OUTPUT_DIR} to exist."
    )


def test_db_directory_exists():
    assert os.path.isdir(DB_URI), (
        f"Expected the LanceDB database directory {DB_URI} to exist before evaluation."
    )


def test_inventory_table_seeded_with_expected_schema():
    """The inventory table must already exist with the original schema and seed rows."""
    import lancedb  # type: ignore
    import pyarrow as pa  # type: ignore

    db = lancedb.connect(DB_URI)
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"Expected table '{TABLE_NAME}' to exist in LanceDB at {DB_URI}. "
        f"Found tables: {table_names}"
    )

    table = db.open_table(TABLE_NAME)
    schema = table.schema
    field_names = {field.name for field in schema}
    expected_initial_fields = {"id", "sku", "qty", "vector"}
    assert field_names == expected_initial_fields, (
        f"Expected initial inventory table fields {expected_initial_fields}, "
        f"got {field_names}."
    )

    id_field = schema.field("id")
    assert pa.types.is_int64(id_field.type), (
        f"Expected initial 'id' column to be int64, got {id_field.type}."
    )

    sku_field = schema.field("sku")
    assert pa.types.is_string(sku_field.type) or pa.types.is_large_string(sku_field.type), (
        f"Expected initial 'sku' column to be a string type, got {sku_field.type}."
    )

    qty_field = schema.field("qty")
    assert pa.types.is_int32(qty_field.type), (
        f"Expected initial 'qty' column to be int32, got {qty_field.type}."
    )

    vector_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type), (
        f"Expected initial 'vector' column to be a fixed-size list, got {vector_field.type}."
    )
    assert vector_field.type.list_size == 4, (
        f"Expected initial 'vector' column to have 4 dimensions, got {vector_field.type.list_size}."
    )

    row_count = table.count_rows()
    assert row_count == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} seeded rows in the inventory table, got {row_count}."
    )


def test_inventory_schema_after_summary_not_yet_present():
    """The candidate's output JSON should not exist before evaluation begins."""
    summary_path = os.path.join(OUTPUT_DIR, "schema_after.json")
    assert not os.path.exists(summary_path), (
        f"Expected {summary_path} to be absent before the candidate runs; "
        "remove any leftover output before evaluation."
    )
