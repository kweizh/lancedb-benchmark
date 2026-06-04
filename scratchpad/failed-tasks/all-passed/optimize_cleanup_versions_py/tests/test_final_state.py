import json
import os

import pytest


OUTPUT_FILE = "/workspace/output/optimize_state.json"
DEFAULT_URI = "/workspace/db"
TABLE_NAME = "metrics"
EXPECTED_ROW_COUNT = 180


@pytest.fixture(scope="module")
def state_payload():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Expected output file {OUTPUT_FILE} does not exist."
    )
    with open(OUTPUT_FILE, "r") as fh:
        raw = fh.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{OUTPUT_FILE} is not valid JSON: {exc}; content was: {raw!r}")
    assert isinstance(payload, dict), (
        f"Expected JSON object in {OUTPUT_FILE}, got type {type(payload).__name__}."
    )
    return payload


def test_output_keys_present_and_int(state_payload):
    required_keys = {"pre_optimize_versions", "post_optimize_versions", "post_optimize_row_count"}
    missing = required_keys - set(state_payload.keys())
    assert not missing, f"Missing required keys in {OUTPUT_FILE}: {sorted(missing)}"
    for key in required_keys:
        value = state_payload[key]
        # JSON does not distinguish int vs bool clearly; explicitly reject bool.
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"Key '{key}' must be an integer, got {type(value).__name__}: {value!r}"
        )


def test_post_optimize_row_count(state_payload):
    assert state_payload["post_optimize_row_count"] == EXPECTED_ROW_COUNT, (
        f"post_optimize_row_count must be {EXPECTED_ROW_COUNT} (100 seed + 8*10 added), "
        f"got {state_payload['post_optimize_row_count']}."
    )


def test_pre_versions_greater_than_post(state_payload):
    pre = state_payload["pre_optimize_versions"]
    post = state_payload["post_optimize_versions"]
    assert pre > post, (
        f"Expected pre_optimize_versions > post_optimize_versions (cleanup should prune "
        f"older versions); got pre={pre}, post={post}."
    )


def test_post_versions_at_least_one(state_payload):
    post = state_payload["post_optimize_versions"]
    assert post >= 1, (
        f"Expected at least 1 version to remain after cleanup (the current version is "
        f"always retained); got post_optimize_versions={post}."
    )


def test_lancedb_store_matches_payload(state_payload):
    import lancedb  # imported lazily so collection doesn't fail before deps install

    uri = os.environ.get("LANCEDB_URI", DEFAULT_URI)
    db = lancedb.connect(uri)
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"Expected table '{TABLE_NAME}' to exist at LanceDB URI {uri}; saw tables: {table_names}."
    )
    table = db.open_table(TABLE_NAME)

    actual_rows = table.count_rows()
    assert actual_rows == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} rows in '{TABLE_NAME}', got {actual_rows}."
    )
    assert actual_rows == state_payload["post_optimize_row_count"], (
        f"Row count in LanceDB ({actual_rows}) does not match post_optimize_row_count "
        f"in {OUTPUT_FILE} ({state_payload['post_optimize_row_count']})."
    )

    versions = table.list_versions()
    assert len(versions) == state_payload["post_optimize_versions"], (
        f"Live version count ({len(versions)}) does not match post_optimize_versions "
        f"in {OUTPUT_FILE} ({state_payload['post_optimize_versions']})."
    )
