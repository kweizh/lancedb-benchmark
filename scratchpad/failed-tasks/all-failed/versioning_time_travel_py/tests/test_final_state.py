import json
import os

import lancedb
import pytest

OUTPUT_PATH = "/workspace/output/version_state.json"
EXPECTED_EARLY_IDS = [1, 2, 3, 4, 5]
EXPECTED_LATEST_IDS = [2, 3, 4, 5, 6, 7, 8]
EXPECTED_LATEST_ID3_TITLE = "v3-updated"
MIN_NUM_VERSIONS = 5


@pytest.fixture(scope="module")
def result_payload():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected output file {OUTPUT_PATH} does not exist."
    )
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"{OUTPUT_PATH} is not valid JSON: {e}")
    assert isinstance(data, dict), (
        f"Top-level JSON in {OUTPUT_PATH} must be an object, got {type(data).__name__}."
    )
    return data


def test_num_versions_is_integer_and_meets_minimum(result_payload):
    value = result_payload.get("num_versions")
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"'num_versions' must be an integer, got {type(value).__name__}: {value!r}."
    )
    assert value >= MIN_NUM_VERSIONS, (
        f"'num_versions' should be >= {MIN_NUM_VERSIONS} after the required mutations, got {value}."
    )


def test_early_version_ids(result_payload):
    value = result_payload.get("early_version_ids_sorted")
    assert isinstance(value, list), (
        f"'early_version_ids_sorted' must be a list, got {type(value).__name__}."
    )
    coerced = [int(v) for v in value]
    assert coerced == EXPECTED_EARLY_IDS, (
        f"'early_version_ids_sorted' should equal {EXPECTED_EARLY_IDS}, got {coerced}."
    )


def test_latest_version_ids(result_payload):
    value = result_payload.get("latest_version_ids_sorted")
    assert isinstance(value, list), (
        f"'latest_version_ids_sorted' must be a list, got {type(value).__name__}."
    )
    coerced = [int(v) for v in value]
    assert coerced == EXPECTED_LATEST_IDS, (
        f"'latest_version_ids_sorted' should equal {EXPECTED_LATEST_IDS}, got {coerced}."
    )


def test_latest_id3_title(result_payload):
    value = result_payload.get("latest_id3_title")
    assert isinstance(value, str), (
        f"'latest_id3_title' must be a string, got {type(value).__name__}."
    )
    assert value == EXPECTED_LATEST_ID3_TITLE, (
        f"'latest_id3_title' should equal {EXPECTED_LATEST_ID3_TITLE!r}, got {value!r}."
    )


def test_table_latest_state_matches_via_sdk():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)
    table_names = list(db.table_names())
    assert "docs" in table_names, (
        f"Expected a 'docs' table in LanceDB at {uri}, found tables: {table_names}."
    )
    table = db.open_table("docs")
    table.checkout_latest()
    rows = table.to_pandas()
    ids_sorted = sorted(int(x) for x in rows["id"].tolist())
    assert ids_sorted == EXPECTED_LATEST_IDS, (
        f"Latest snapshot of 'docs' should contain ids {EXPECTED_LATEST_IDS}, got {ids_sorted}."
    )
    id3_rows = rows[rows["id"] == 3]
    assert len(id3_rows) == 1, (
        f"Expected exactly one row with id=3 in the latest snapshot, found {len(id3_rows)}."
    )
    assert id3_rows.iloc[0]["title"] == EXPECTED_LATEST_ID3_TITLE, (
        f"Expected id=3 title to be {EXPECTED_LATEST_ID3_TITLE!r}, got "
        f"{id3_rows.iloc[0]['title']!r}."
    )
    versions = table.list_versions()
    assert len(versions) >= MIN_NUM_VERSIONS, (
        f"Expected at least {MIN_NUM_VERSIONS} versions on 'docs', got {len(versions)}."
    )
