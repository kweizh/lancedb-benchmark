import json
import math
import os

import numpy as np
import pytest


LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_FILE = "/workspace/output/upsert_state.json"
TABLE_NAME = "users"

UPDATED_IDS = (2, 5, 7)
INSERTED_IDS = (11, 12)
UNCHANGED_IDS = (1, 10)
EXPECTED_JSON_IDS = [1, 2, 5, 7, 10, 11, 12]


@pytest.fixture(scope="module")
def lance_table():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    assert TABLE_NAME in db.table_names(), (
        f"Table '{TABLE_NAME}' not found in LanceDB at {LANCEDB_URI}. "
        f"Available tables: {db.table_names()}"
    )
    return db.open_table(TABLE_NAME)


@pytest.fixture(scope="module")
def lance_rows(lance_table):
    df = lance_table.to_pandas()
    rows = {int(row["id"]): row for _, row in df.iterrows()}
    return rows


@pytest.fixture(scope="module")
def output_payload():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Expected output file {OUTPUT_FILE} to exist after the candidate's script runs."
    )
    with open(OUTPUT_FILE, "r") as fh:
        payload = json.load(fh)
    return payload


@pytest.fixture(scope="module")
def expected_seed_scores():
    return np.random.default_rng(0).random(10).astype("float32")


def test_row_count(lance_table):
    count = lance_table.count_rows()
    assert count == 12, f"Expected count_rows() == 12 after upsert, got {count}."


def test_output_is_list_of_seven(output_payload):
    assert isinstance(output_payload, list), (
        f"Output JSON must be a list, got {type(output_payload).__name__}."
    )
    assert len(output_payload) == 7, (
        f"Expected 7 rows in output JSON, got {len(output_payload)}."
    )


def test_output_sort_order(output_payload):
    ids = [item["id"] for item in output_payload]
    assert ids == EXPECTED_JSON_IDS, (
        f"Output JSON ids must be sorted ascending as {EXPECTED_JSON_IDS}, got {ids}."
    )


def test_output_schema(output_payload):
    for item in output_payload:
        assert set(item.keys()) >= {"id", "email", "score"}, (
            f"Each output entry must contain 'id', 'email', 'score'. Got: {list(item.keys())}"
        )
        assert isinstance(item["id"], int), f"'id' must be int, got {type(item['id']).__name__}: {item}"
        assert isinstance(item["email"], str), f"'email' must be string, got {type(item['email']).__name__}: {item}"
        assert isinstance(item["score"], (int, float)), (
            f"'score' must be a number, got {type(item['score']).__name__}: {item}"
        )


@pytest.mark.parametrize("id_value", UPDATED_IDS)
def test_updated_rows_in_lancedb(lance_rows, id_value):
    assert id_value in lance_rows, f"Updated row id={id_value} missing from LanceDB."
    row = lance_rows[id_value]
    expected_email = f"updated_{id_value}@example.com"
    expected_score = 0.5 + 0.1 * id_value
    assert row["email"] == expected_email, (
        f"LanceDB row id={id_value} email should be '{expected_email}', got '{row['email']}'."
    )
    assert math.isclose(float(row["score"]), expected_score, abs_tol=1e-4), (
        f"LanceDB row id={id_value} score should be ~{expected_score}, got {float(row['score'])}."
    )


@pytest.mark.parametrize("id_value", UPDATED_IDS)
def test_updated_rows_in_output(output_payload, id_value):
    item = next((it for it in output_payload if it["id"] == id_value), None)
    assert item is not None, f"Output JSON missing updated id={id_value}."
    assert item["email"] == f"updated_{id_value}@example.com", (
        f"Output id={id_value} email should be 'updated_{id_value}@example.com', got '{item['email']}'."
    )
    assert math.isclose(float(item["score"]), 0.5 + 0.1 * id_value, abs_tol=1e-4), (
        f"Output id={id_value} score should be ~{0.5 + 0.1 * id_value}, got {item['score']}."
    )


@pytest.mark.parametrize("id_value", INSERTED_IDS)
def test_inserted_rows_in_lancedb(lance_rows, id_value):
    assert id_value in lance_rows, f"Inserted row id={id_value} missing from LanceDB."
    row = lance_rows[id_value]
    expected_email = f"new_{id_value}@example.com"
    expected_score = 0.5 + 0.1 * id_value
    assert row["email"] == expected_email, (
        f"LanceDB row id={id_value} email should be '{expected_email}', got '{row['email']}'."
    )
    assert math.isclose(float(row["score"]), expected_score, abs_tol=1e-4), (
        f"LanceDB row id={id_value} score should be ~{expected_score}, got {float(row['score'])}."
    )


@pytest.mark.parametrize("id_value", INSERTED_IDS)
def test_inserted_rows_in_output(output_payload, id_value):
    item = next((it for it in output_payload if it["id"] == id_value), None)
    assert item is not None, f"Output JSON missing inserted id={id_value}."
    assert item["email"] == f"new_{id_value}@example.com", (
        f"Output id={id_value} email should be 'new_{id_value}@example.com', got '{item['email']}'."
    )
    assert math.isclose(float(item["score"]), 0.5 + 0.1 * id_value, abs_tol=1e-4), (
        f"Output id={id_value} score should be ~{0.5 + 0.1 * id_value}, got {item['score']}."
    )


@pytest.mark.parametrize("id_value", UNCHANGED_IDS)
def test_unchanged_rows_in_lancedb(lance_rows, expected_seed_scores, id_value):
    assert id_value in lance_rows, f"Unchanged row id={id_value} missing from LanceDB."
    row = lance_rows[id_value]
    expected_email = f"user_{id_value}@example.com"
    expected_score = float(expected_seed_scores[id_value - 1])
    assert row["email"] == expected_email, (
        f"LanceDB row id={id_value} email should be '{expected_email}', got '{row['email']}'."
    )
    assert math.isclose(float(row["score"]), expected_score, abs_tol=1e-4), (
        f"LanceDB row id={id_value} score should be ~{expected_score} (seed RNG), got {float(row['score'])}."
    )


@pytest.mark.parametrize("id_value", UNCHANGED_IDS)
def test_unchanged_rows_in_output(output_payload, expected_seed_scores, id_value):
    item = next((it for it in output_payload if it["id"] == id_value), None)
    assert item is not None, f"Output JSON missing unchanged id={id_value}."
    expected_email = f"user_{id_value}@example.com"
    expected_score = float(expected_seed_scores[id_value - 1])
    assert item["email"] == expected_email, (
        f"Output id={id_value} email should be '{expected_email}', got '{item['email']}'."
    )
    assert math.isclose(float(item["score"]), expected_score, abs_tol=1e-4), (
        f"Output id={id_value} score should be ~{expected_score}, got {item['score']}."
    )
