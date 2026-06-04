import json
import os

import pytest


OUTPUT_PATH = "/workspace/output/counts.json"
DEFAULT_LANCEDB_URI = "/workspace/db"
TABLE_NAME = "orders"
EXPECTED_COUNTS = {
    "total_rows": 60,
    "us_orders": 20,
    "unpaid_high_value": 21,
    "apac_or_eu": 40,
}


def _lancedb_uri():
    return os.environ.get("LANCEDB_URI", DEFAULT_LANCEDB_URI)


@pytest.fixture(scope="module")
def counts_payload():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected output JSON at {OUTPUT_PATH} to exist after the task runs."
    )
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{OUTPUT_PATH} is not valid JSON: {exc}")
    assert isinstance(data, dict), (
        f"Expected {OUTPUT_PATH} to contain a JSON object, got {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def orders_table():
    import lancedb

    uri = _lancedb_uri()
    db = lancedb.connect(uri)
    table_names = db.table_names()
    assert TABLE_NAME in table_names, (
        f"Expected table {TABLE_NAME!r} to exist in LanceDB at {uri}, found: {table_names}."
    )
    return db.open_table(TABLE_NAME)


def test_counts_json_keys(counts_payload):
    assert set(counts_payload.keys()) == set(EXPECTED_COUNTS.keys()), (
        f"Expected exactly keys {sorted(EXPECTED_COUNTS.keys())} in counts.json, "
        f"got {sorted(counts_payload.keys())}."
    )


def test_counts_json_values_are_ints(counts_payload):
    for key in EXPECTED_COUNTS:
        value = counts_payload[key]
        # Booleans are a subclass of int — reject them explicitly.
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"Expected counts.json[{key!r}] to be an integer, got {type(value).__name__}: {value!r}."
        )


def test_counts_json_matches_expected_values(counts_payload):
    for key, expected in EXPECTED_COUNTS.items():
        assert counts_payload[key] == expected, (
            f"Expected counts.json[{key!r}] == {expected}, got {counts_payload[key]!r}."
        )


def test_orders_table_schema(orders_table):
    schema = orders_table.schema
    field_names = {f.name for f in schema}
    required = {"id", "region", "amount", "paid", "vector"}
    missing = required - field_names
    assert not missing, (
        f"Expected table {TABLE_NAME!r} to contain fields {sorted(required)}, "
        f"missing: {sorted(missing)}. Actual schema: {schema}."
    )

    fields_by_name = {f.name: f for f in schema}

    import pyarrow as pa

    assert pa.types.is_int64(fields_by_name["id"].type), (
        f"Expected 'id' to be int64, got {fields_by_name['id'].type}."
    )
    assert pa.types.is_string(fields_by_name["region"].type) or pa.types.is_large_string(
        fields_by_name["region"].type
    ), f"Expected 'region' to be a string type, got {fields_by_name['region'].type}."
    assert pa.types.is_float64(fields_by_name["amount"].type), (
        f"Expected 'amount' to be float64, got {fields_by_name['amount'].type}."
    )
    assert pa.types.is_boolean(fields_by_name["paid"].type), (
        f"Expected 'paid' to be bool, got {fields_by_name['paid'].type}."
    )
    vector_type = fields_by_name["vector"].type
    assert pa.types.is_fixed_size_list(vector_type), (
        f"Expected 'vector' to be a fixed_size_list, got {vector_type}."
    )
    assert vector_type.list_size == 4, (
        f"Expected 'vector' fixed_size_list length 4, got {vector_type.list_size}."
    )
    assert pa.types.is_float32(vector_type.value_type), (
        f"Expected 'vector' element type to be float32, got {vector_type.value_type}."
    )


def test_orders_table_total_rows(orders_table):
    total = orders_table.count_rows()
    assert total == 60, f"Expected 60 rows in {TABLE_NAME!r}, got {total}."


def test_orders_table_us_filter(orders_table):
    n = orders_table.count_rows(filter="region = 'us'")
    assert n == 20, (
        f"Expected count_rows(filter=\"region = 'us'\") == 20, got {n}."
    )


def test_orders_table_unpaid_high_value_filter(orders_table):
    n = orders_table.count_rows(filter="paid = false AND amount > 100.0")
    assert n == 21, (
        "Expected count_rows(filter=\"paid = false AND amount > 100.0\") == 21, "
        f"got {n}."
    )


def test_orders_table_apac_or_eu_filter(orders_table):
    n = orders_table.count_rows(filter="region IN ('eu', 'apac')")
    assert n == 40, (
        "Expected count_rows(filter=\"region IN ('eu', 'apac')\") == 40, "
        f"got {n}."
    )
