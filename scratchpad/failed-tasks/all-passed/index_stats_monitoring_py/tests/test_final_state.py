import json
import os

import lancedb

ARTIFACT_PATH = "/workspace/output/index_stats.json"
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "points"
INDEX_NAME = "vector_idx"
EXPECTED_KEYS = {"index_type", "initial_indexed", "initial_unindexed", "unindexed_after_add"}


def _load_artifact():
    assert os.path.isfile(ARTIFACT_PATH), (
        f"Expected output artifact {ARTIFACT_PATH} to exist after running the task."
    )
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_artifact_exists_and_is_json():
    data = _load_artifact()
    assert isinstance(data, dict), (
        f"Expected {ARTIFACT_PATH} to contain a JSON object, got {type(data).__name__}."
    )


def test_artifact_has_exact_keys():
    data = _load_artifact()
    actual = set(data.keys())
    assert actual == EXPECTED_KEYS, (
        f"Expected top-level keys {sorted(EXPECTED_KEYS)}, got {sorted(actual)}."
    )


def test_index_type_is_ivf_pq():
    data = _load_artifact()
    index_type = data["index_type"]
    assert isinstance(index_type, str), (
        f"index_type must be a string, got {type(index_type).__name__}."
    )
    assert "IVF_PQ" in index_type.upper(), (
        f"Expected index_type to contain 'IVF_PQ' (case-insensitive), got {index_type!r}."
    )


def test_initial_indexed_at_least_256():
    data = _load_artifact()
    initial_indexed = data["initial_indexed"]
    assert isinstance(initial_indexed, int) and not isinstance(initial_indexed, bool), (
        f"initial_indexed must be an int, got {type(initial_indexed).__name__}."
    )
    assert initial_indexed >= 256, (
        f"Expected initial_indexed >= 256 after building IVF_PQ on 400 rows, got {initial_indexed}."
    )


def test_initial_unindexed_is_zero():
    data = _load_artifact()
    initial_unindexed = data["initial_unindexed"]
    assert isinstance(initial_unindexed, int) and not isinstance(initial_unindexed, bool), (
        f"initial_unindexed must be an int, got {type(initial_unindexed).__name__}."
    )
    assert initial_unindexed == 0, (
        f"Expected initial_unindexed == 0 immediately after wait_for_index, got {initial_unindexed}."
    )


def test_unindexed_after_add_at_least_50():
    data = _load_artifact()
    unindexed_after_add = data["unindexed_after_add"]
    assert isinstance(unindexed_after_add, int) and not isinstance(unindexed_after_add, bool), (
        f"unindexed_after_add must be an int, got {type(unindexed_after_add).__name__}."
    )
    assert unindexed_after_add >= 50, (
        f"Expected unindexed_after_add >= 50 after appending 50 rows, got {unindexed_after_add}."
    )


def test_table_has_450_rows_in_lancedb():
    db = lancedb.connect(LANCEDB_URI)
    assert TABLE_NAME in db.table_names(), (
        f"Expected table {TABLE_NAME!r} to exist at {LANCEDB_URI}."
    )
    table = db.open_table(TABLE_NAME)
    total = table.count_rows()
    assert total == 450, (
        f"Expected 450 rows (400 seed + 50 added) in table {TABLE_NAME!r}, got {total}."
    )


def test_live_index_stats_show_unindexed_delta():
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(TABLE_NAME)
    stats = table.index_stats(INDEX_NAME)
    assert stats is not None, (
        f"Expected index_stats({INDEX_NAME!r}) to return an object, got None."
    )
    num_unindexed = int(getattr(stats, "num_unindexed_rows", 0))
    assert num_unindexed >= 50, (
        f"Expected live num_unindexed_rows >= 50 (no optimize() between snapshots), got {num_unindexed}."
    )
    index_type = str(getattr(stats, "index_type", ""))
    assert "IVF_PQ" in index_type.upper(), (
        f"Expected live index_type to contain 'IVF_PQ', got {index_type!r}."
    )
