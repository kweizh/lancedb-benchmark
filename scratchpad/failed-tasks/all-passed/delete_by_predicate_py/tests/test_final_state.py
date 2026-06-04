import json
import os

import lancedb
import pytest


OUTPUT_FILE = "/workspace/output/delete_state.json"
EXPECTED_REMAINING_IDS = [
    1, 3, 4, 6, 7, 10, 12, 15, 16, 18, 19, 21,
    22, 24, 25, 27, 28, 30, 33, 36, 39, 42, 45, 48,
]
EXPECTED_TOTAL_ROWS = 24


def _db_uri() -> str:
    return os.environ.get("LANCEDB_URI", "/workspace/db")


@pytest.fixture(scope="module")
def loaded_state():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Expected output JSON file at {OUTPUT_FILE} to exist after the task runs."
    )
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def logs_table():
    db = lancedb.connect(_db_uri())
    table_names = list(db.table_names())
    assert "logs" in table_names, (
        f"Expected a table named 'logs' in LanceDB at {_db_uri()}, "
        f"but found: {table_names}"
    )
    return db.open_table("logs")


def test_output_json_has_required_keys(loaded_state):
    assert isinstance(loaded_state, dict), (
        "Expected the output JSON top-level value to be a JSON object."
    )
    assert "total_rows" in loaded_state, (
        "Expected key 'total_rows' in delete_state.json."
    )
    assert "remaining_ids_sorted" in loaded_state, (
        "Expected key 'remaining_ids_sorted' in delete_state.json."
    )


def test_output_total_rows_matches_expected(loaded_state):
    total = loaded_state["total_rows"]
    assert isinstance(total, int), (
        f"Expected 'total_rows' to be an integer, got {type(total).__name__}: {total!r}"
    )
    assert total == EXPECTED_TOTAL_ROWS, (
        f"Expected 'total_rows' to be {EXPECTED_TOTAL_ROWS}, got {total}."
    )


def test_output_remaining_ids_match_expected(loaded_state):
    remaining = loaded_state["remaining_ids_sorted"]
    assert isinstance(remaining, list), (
        f"Expected 'remaining_ids_sorted' to be a list, got {type(remaining).__name__}."
    )
    assert all(isinstance(x, int) for x in remaining), (
        "Expected every element of 'remaining_ids_sorted' to be an integer."
    )
    assert remaining == EXPECTED_REMAINING_IDS, (
        f"Expected 'remaining_ids_sorted' to equal {EXPECTED_REMAINING_IDS}, got {remaining}."
    )


def test_table_count_rows_matches_expected(logs_table):
    count = logs_table.count_rows()
    assert count == EXPECTED_TOTAL_ROWS, (
        f"Expected logs table to have {EXPECTED_TOTAL_ROWS} rows after deletes, got {count}."
    )


def test_table_remaining_ids_match_expected(logs_table):
    df = logs_table.to_pandas()
    ids_sorted = sorted(int(x) for x in df["id"].tolist())
    assert ids_sorted == EXPECTED_REMAINING_IDS, (
        f"Expected sorted ids in logs table to equal {EXPECTED_REMAINING_IDS}, got {ids_sorted}."
    )


def test_no_warn_rows_remain(logs_table):
    df = logs_table.to_pandas()
    warn_rows = df[df["level"] == "warn"]
    assert len(warn_rows) == 0, (
        f"Expected zero rows with level='warn' after deletes, found {len(warn_rows)}: "
        f"{warn_rows.to_dict(orient='records')}"
    )


def test_no_info_seq_gt_60_rows_remain(logs_table):
    df = logs_table.to_pandas()
    info_high = df[(df["level"] == "info") & (df["seq"] > 60)]
    assert len(info_high) == 0, (
        "Expected zero rows with level='info' AND seq > 60 after deletes, "
        f"found {len(info_high)}: {info_high.to_dict(orient='records')}"
    )


def test_explicit_id_deletes_applied(logs_table):
    df = logs_table.to_pandas()
    remaining_ids = set(int(x) for x in df["id"].tolist())
    for forbidden_id in (5, 9, 13):
        assert forbidden_id not in remaining_ids, (
            f"Expected id={forbidden_id} to be removed by the explicit IN-list delete, "
            f"but it is still present."
        )
