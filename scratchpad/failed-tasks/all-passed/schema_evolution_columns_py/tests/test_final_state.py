import importlib
import json
import os

import pytest


WORKSPACE = "/workspace"
SOLUTION_DIR = "/workspace/solution"
OUTPUT_DIR = "/workspace/output"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "schema_after.json")
DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "inventory"

# Deterministic seed for the inventory fixture (must match the Dockerfile seeder).
SEED_QTY = [3, 12, 1, 7, 8, 4, 15, 2, 10, 5, 0, 6]
SEED_IDS = list(range(1, len(SEED_QTY) + 1))
SEED_SKUS = [f"SKU-{i:03d}" for i in SEED_IDS]

EXPECTED_FIELD_NAMES_SORTED = ["id", "is_low_stock", "price_cents", "qty", "sku"]
EXPECTED_LOW_STOCK_IDS = sorted(
    id_ for id_, q in zip(SEED_IDS, SEED_QTY) if q < 5
)
EXPECTED_TOTAL_PRICE_CENTS = sum(SEED_QTY) * 100
EXPECTED_ROW_COUNT = len(SEED_QTY)


@pytest.fixture(scope="module")
def opened_table():
    lancedb = importlib.import_module("lancedb")
    db = lancedb.connect(DB_URI)
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"Expected table '{TABLE_NAME}' to exist in LanceDB at {DB_URI}. "
        f"Found tables: {table_names}"
    )
    return db.open_table(TABLE_NAME)


def test_summary_file_exists_and_is_valid_json():
    assert os.path.isfile(SUMMARY_PATH), (
        f"Expected output JSON file at {SUMMARY_PATH} but it was not found."
    )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"Expected the JSON document at {SUMMARY_PATH} to be a JSON object, "
        f"got {type(data).__name__}."
    )


def test_summary_field_names_sorted():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "field_names_sorted" in data, (
        f"Expected 'field_names_sorted' key in {SUMMARY_PATH}, got keys: {list(data.keys())}."
    )
    field_names_sorted = data["field_names_sorted"]
    assert isinstance(field_names_sorted, list), (
        "'field_names_sorted' must be a JSON array of column names."
    )
    assert field_names_sorted == EXPECTED_FIELD_NAMES_SORTED, (
        f"Expected field_names_sorted={EXPECTED_FIELD_NAMES_SORTED}, "
        f"got {field_names_sorted}."
    )


def test_summary_low_stock_ids_sorted():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "low_stock_ids_sorted" in data, (
        f"Expected 'low_stock_ids_sorted' key in {SUMMARY_PATH}, got keys: {list(data.keys())}."
    )
    low_stock_ids_sorted = data["low_stock_ids_sorted"]
    assert isinstance(low_stock_ids_sorted, list), (
        "'low_stock_ids_sorted' must be a JSON array of integers."
    )
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in low_stock_ids_sorted), (
        "'low_stock_ids_sorted' must contain only integer values."
    )
    assert low_stock_ids_sorted == EXPECTED_LOW_STOCK_IDS, (
        f"Expected low_stock_ids_sorted={EXPECTED_LOW_STOCK_IDS}, got {low_stock_ids_sorted}."
    )


def test_summary_total_price_cents():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "total_price_cents" in data, (
        f"Expected 'total_price_cents' key in {SUMMARY_PATH}, got keys: {list(data.keys())}."
    )
    total_price_cents = data["total_price_cents"]
    assert isinstance(total_price_cents, int) and not isinstance(total_price_cents, bool), (
        "'total_price_cents' must be a JSON integer."
    )
    assert total_price_cents == EXPECTED_TOTAL_PRICE_CENTS, (
        f"Expected total_price_cents={EXPECTED_TOTAL_PRICE_CENTS}, got {total_price_cents}."
    )


def test_table_schema_evolved(opened_table):
    import pyarrow as pa  # type: ignore

    schema = opened_table.schema
    field_names = {field.name for field in schema}
    expected_field_names = {"id", "sku", "qty", "price_cents", "is_low_stock"}
    assert field_names == expected_field_names, (
        f"Expected post-evolution fields {expected_field_names}, got {field_names}."
    )
    assert "vector" not in field_names, (
        "Expected the 'vector' column to be dropped from the inventory table."
    )

    price_field = schema.field("price_cents")
    assert pa.types.is_integer(price_field.type) and price_field.type.bit_width == 64, (
        f"Expected 'price_cents' to be a 64-bit integer column, got {price_field.type}."
    )

    flag_field = schema.field("is_low_stock")
    assert pa.types.is_boolean(flag_field.type), (
        f"Expected 'is_low_stock' to be a boolean column, got {flag_field.type}."
    )


def test_table_row_count_preserved(opened_table):
    row_count = opened_table.count_rows()
    assert row_count == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} rows after schema evolution, got {row_count}."
    )


def test_derived_columns_match_seed(opened_table):
    df = opened_table.to_pandas()
    assert set(df.columns) >= {"id", "qty", "price_cents", "is_low_stock"}, (
        f"Expected derived columns to be readable, got columns: {list(df.columns)}."
    )

    df = df.sort_values("id").reset_index(drop=True)
    ids = df["id"].tolist()
    qtys = df["qty"].tolist()
    price_cents = df["price_cents"].tolist()
    is_low_stock = df["is_low_stock"].tolist()

    assert ids == SEED_IDS, f"Expected seeded ids {SEED_IDS}, got {ids}."
    assert qtys == SEED_QTY, f"Expected seeded qty {SEED_QTY}, got {qtys}."

    expected_price_cents = [q * 100 for q in SEED_QTY]
    assert [int(x) for x in price_cents] == expected_price_cents, (
        f"Expected price_cents={expected_price_cents}, got {price_cents}."
    )

    expected_low_stock = [q < 5 for q in SEED_QTY]
    assert [bool(x) for x in is_low_stock] == expected_low_stock, (
        f"Expected is_low_stock={expected_low_stock}, got {is_low_stock}."
    )
