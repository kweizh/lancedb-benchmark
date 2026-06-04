import json
import os
from datetime import datetime, timedelta

import lancedb
import pytest

OUTPUT_PATH = "/workspace/output/filter_results.json"

EVENT_TYPE_CYCLE = ["info", "warn", "error", "warn", "info", "error"]
SEVERITY_CYCLE = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
NUM_ROWS = 30


def _build_fixture():
    rows = []
    base = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(NUM_ROWS):
        rows.append(
            {
                "id": i,
                "event_type": EVENT_TYPE_CYCLE[i % len(EVENT_TYPE_CYCLE)],
                "severity": SEVERITY_CYCLE[i % len(SEVERITY_CYCLE)],
                "created_at": base + timedelta(hours=i),
            }
        )
    return rows


def _ground_truth():
    rows = _build_fixture()
    high = sorted(
        ({"id": r["id"], "event_type": r["event_type"], "severity": r["severity"]} for r in rows if r["severity"] >= 7),
        key=lambda r: r["id"],
    )
    errors = sorted(
        (
            {"id": r["id"], "event_type": r["event_type"], "severity": r["severity"]}
            for r in rows
            if r["event_type"] == "error"
        ),
        key=lambda r: r["id"],
    )
    combined = sorted(
        (
            {"id": r["id"], "event_type": r["event_type"], "severity": r["severity"]}
            for r in rows
            if r["event_type"] == "warn" and r["severity"] >= 3
        ),
        key=lambda r: r["id"],
    )
    return {"high_severity": high, "errors_only": errors, "combined": combined}


@pytest.fixture(scope="module")
def loaded_results():
    assert os.path.isfile(OUTPUT_PATH), f"Output file {OUTPUT_PATH} does not exist."
    with open(OUTPUT_PATH, "r") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{OUTPUT_PATH} is not valid JSON: {exc}")
    return data


def test_output_file_top_level_keys(loaded_results):
    assert isinstance(loaded_results, dict), (
        f"Top-level JSON value must be an object, got {type(loaded_results).__name__}."
    )
    expected_keys = {"high_severity", "errors_only", "combined"}
    actual_keys = set(loaded_results.keys())
    assert actual_keys == expected_keys, (
        f"Top-level keys must be exactly {sorted(expected_keys)}, got {sorted(actual_keys)}."
    )


def _normalize_row(row, key):
    assert isinstance(row, dict), f"Each entry in '{key}' must be a JSON object, got {type(row).__name__}."
    assert set(row.keys()) == {"id", "event_type", "severity"}, (
        f"Each entry in '{key}' must have exactly the keys id, event_type, severity. "
        f"Got {sorted(row.keys())}."
    )
    return {"id": int(row["id"]), "event_type": str(row["event_type"]), "severity": int(row["severity"])}


@pytest.mark.parametrize("key", ["high_severity", "errors_only", "combined"])
def test_filter_results_match_ground_truth(loaded_results, key):
    expected = _ground_truth()[key]
    actual_raw = loaded_results.get(key)
    assert isinstance(actual_raw, list), f"Value for '{key}' must be a JSON array, got {type(actual_raw).__name__}."

    actual = [_normalize_row(row, key) for row in actual_raw]

    actual_ids = [row["id"] for row in actual]
    assert actual_ids == sorted(actual_ids), (
        f"Results for '{key}' must be sorted by id ascending. Got id order {actual_ids}."
    )

    assert actual == expected, (
        f"Results for '{key}' do not match ground truth.\nExpected: {expected}\nActual:   {actual}"
    )


def test_events_table_matches_fixture():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)

    table_names = list(db.table_names())
    assert "events" in table_names, (
        f"LanceDB at {uri} must contain a table named 'events'. Found: {table_names}."
    )

    tbl = db.open_table("events")
    pa_table = tbl.to_arrow()

    assert pa_table.num_rows == NUM_ROWS, (
        f"'events' table must have exactly {NUM_ROWS} rows, found {pa_table.num_rows}."
    )

    column_names = set(pa_table.column_names)
    expected_columns = {"id", "event_type", "severity", "created_at", "vector"}
    assert expected_columns.issubset(column_names), (
        f"'events' table is missing required columns. Expected superset of {sorted(expected_columns)}, "
        f"got {sorted(column_names)}."
    )

    df = pa_table.select(["id", "event_type", "severity"]).to_pandas().sort_values("id").reset_index(drop=True)
    fixture = _build_fixture()
    for i, row in df.iterrows():
        assert int(row["id"]) == fixture[i]["id"], (
            f"Row {i} id mismatch: got {row['id']}, expected {fixture[i]['id']}."
        )
        assert str(row["event_type"]) == fixture[i]["event_type"], (
            f"Row {i} event_type mismatch at id={fixture[i]['id']}: "
            f"got {row['event_type']!r}, expected {fixture[i]['event_type']!r}."
        )
        assert int(row["severity"]) == fixture[i]["severity"], (
            f"Row {i} severity mismatch at id={fixture[i]['id']}: "
            f"got {row['severity']}, expected {fixture[i]['severity']}."
        )
