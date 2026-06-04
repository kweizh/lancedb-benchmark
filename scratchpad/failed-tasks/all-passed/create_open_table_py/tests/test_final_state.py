import json
import os

import pytest


OUTPUT_FILE = "/workspace/output/table_state.json"
EXPECTED_FIELDS = {"id", "name", "price", "tags", "vector"}


def _lancedb_uri() -> str:
    return os.environ.get("LANCEDB_URI", "/workspace/db")


@pytest.fixture(scope="module")
def summary_json():
    assert os.path.isfile(OUTPUT_FILE), f"Expected summary file {OUTPUT_FILE} to exist after the task runs."
    with open(OUTPUT_FILE) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{OUTPUT_FILE} is not valid JSON: {exc!r}")
    assert isinstance(data, dict), f"Expected top-level JSON object in {OUTPUT_FILE}, got {type(data).__name__}."
    return data


def test_summary_has_required_keys(summary_json):
    for key in ("tables_in_db", "row_count", "schema_field_names"):
        assert key in summary_json, f"Summary JSON missing required key '{key}'."


def test_tables_in_db_sorted_and_contains_products(summary_json):
    tables = summary_json["tables_in_db"]
    assert isinstance(tables, list), f"'tables_in_db' should be a list, got {type(tables).__name__}."
    assert all(isinstance(t, str) for t in tables), "'tables_in_db' must contain only strings."
    assert tables == sorted(tables), f"'tables_in_db' must be sorted lexicographically, got {tables}."
    assert "products" in tables, f"'products' must appear in tables_in_db, got {tables}."


def test_row_count_is_six(summary_json):
    row_count = summary_json["row_count"]
    assert isinstance(row_count, int) and not isinstance(row_count, bool), (
        f"'row_count' should be an integer, got {type(row_count).__name__}."
    )
    assert row_count == 6, f"Expected row_count == 6, got {row_count}."


def test_schema_field_names_sorted_and_superset(summary_json):
    fields = summary_json["schema_field_names"]
    assert isinstance(fields, list), f"'schema_field_names' should be a list, got {type(fields).__name__}."
    assert all(isinstance(f, str) for f in fields), "'schema_field_names' must contain only strings."
    assert fields == sorted(fields), f"'schema_field_names' must be sorted, got {fields}."
    missing = EXPECTED_FIELDS - set(fields)
    assert not missing, f"'schema_field_names' is missing required fields: {sorted(missing)}."


def test_products_table_via_sdk():
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(_lancedb_uri())
    names = list(db.table_names())
    assert "products" in names, f"'products' table not found in LanceDB at {_lancedb_uri()}. Tables: {names}."

    tbl = db.open_table("products")
    row_count = tbl.count_rows()
    assert row_count == 6, f"Expected 6 rows in 'products', got {row_count}."

    schema = tbl.schema
    field_names = {f.name for f in schema}
    missing = EXPECTED_FIELDS - field_names
    assert not missing, f"'products' schema is missing required fields: {sorted(missing)}. Got: {sorted(field_names)}."

    type_map = {f.name: f.type for f in schema}

    assert pa.types.is_int32(type_map["id"]), f"'id' should be int32, got {type_map['id']}."
    assert pa.types.is_string(type_map["name"]) or pa.types.is_large_string(type_map["name"]), (
        f"'name' should be a string type, got {type_map['name']}."
    )
    assert pa.types.is_float64(type_map["price"]), f"'price' should be float64, got {type_map['price']}."

    tags_type = type_map["tags"]
    assert pa.types.is_list(tags_type) or pa.types.is_large_list(tags_type), (
        f"'tags' should be a list type, got {tags_type}."
    )
    value_type = tags_type.value_type
    assert pa.types.is_string(value_type) or pa.types.is_large_string(value_type), (
        f"'tags' element type should be string, got {value_type}."
    )

    vector_type = type_map["vector"]
    assert pa.types.is_fixed_size_list(vector_type), (
        f"'vector' should be fixed_size_list, got {vector_type}."
    )
    assert vector_type.list_size == 4, f"'vector' should be a 4-dim fixed_size_list, got size {vector_type.list_size}."
    assert pa.types.is_float32(vector_type.value_type), (
        f"'vector' element type should be float32, got {vector_type.value_type}."
    )
