import importlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest


PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
EXPECTED_PATH = os.path.join(PROJECT_DIR, ".expected.json")
SEED_SCRIPT = "/opt/seed_state.py"
LANCEDB_URI = "/workspace/db"
VECTORS_TABLE = "vectors"
LOG_TABLE = "index_build_log"
SEED_ROW_COUNT = 100
BATCH_SIZE = 50
TARGET_TOTAL = 1050  # > 1024
EXPECTED_BUILDS = 3
THRESHOLDS = [256, 512, 1024]


@pytest.fixture(scope="session")
def reset_state():
    """Wipe LanceDB state and re-run the seed script to restore the starting baseline."""
    # Re-run the seed script to ensure the table is at the 100-row baseline with no index.
    result = subprocess.run(
        ["python3", SEED_SCRIPT],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, (
        f"Seed reset failed (returncode={result.returncode}). "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    return True


@pytest.fixture(scope="session")
def expected_fixture(reset_state):
    assert os.path.isfile(EXPECTED_PATH), (
        f"Expected fixture {EXPECTED_PATH} must exist after the seed reset."
    )
    with open(EXPECTED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def solution_module(reset_state):
    assert os.path.isfile(SOLUTION_PATH), (
        f"Executor must create {SOLUTION_PATH}; not found."
    )
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


@pytest.fixture(scope="session")
def workflow_result(solution_module, expected_fixture):
    """Drive the full workflow exactly once and cache results for the test functions."""
    assert hasattr(solution_module, "get_indexed_table"), (
        "solution.py must expose a zero-argument factory function `get_indexed_table`."
    )
    assert hasattr(solution_module, "IndexedTable"), (
        "solution.py must expose a class named `IndexedTable`."
    )

    t = solution_module.get_indexed_table()
    assert isinstance(t, solution_module.IndexedTable), (
        "get_indexed_table() must return an IndexedTable instance."
    )

    sentinel_id = expected_fixture["sentinel_id"]
    query_vector = expected_fixture["query_vector"]

    # Brute-force sanity check before any threshold is crossed.
    pre_index_top = t.search(query_vector, 5)
    assert isinstance(pre_index_top, list) and len(pre_index_top) == 5, (
        "search(vec, k=5) on the 100-row table must return a list of length 5."
    )
    assert isinstance(pre_index_top[0], dict) and "id" in pre_index_top[0], (
        "Each search result must be a dict containing an `id` key."
    )
    pre_top_id = int(pre_index_top[0]["id"])
    assert pre_top_id == sentinel_id, (
        f"Brute-force search must return sentinel id={sentinel_id} as top-1, got {pre_top_id}."
    )

    # Drive add_rows in 50-row batches from rng(9999).
    rng = np.random.default_rng(9999)
    next_id = 1000  # Avoid colliding with seeded ids (which use 0..99).
    while True:
        # Re-open the table fresh inside this loop so we can check row count via a
        # second connection. (We rely on the implementation's internal count.)
        batch = []
        for _ in range(BATCH_SIZE):
            vec = rng.standard_normal(64).astype(np.float32).tolist()
            batch.append({"id": int(next_id), "vector": vec})
            next_id += 1
        t.add_rows(batch)
        # Inspect external row count to decide when to stop.
        import lancedb

        db_inspect = lancedb.connect(LANCEDB_URI)
        current_count = db_inspect.open_table(VECTORS_TABLE).count_rows()
        if current_count >= TARGET_TOTAL:
            break

    # Final indexed search.
    post_index_top = t.search(query_vector, 5)
    return {
        "table_instance": t,
        "sentinel_id": sentinel_id,
        "query_vector": query_vector,
        "pre_index_top": pre_index_top,
        "post_index_top": post_index_top,
        "final_count": current_count,
    }


def test_solution_module_importable(solution_module):
    assert hasattr(solution_module, "IndexedTable"), (
        "solution.py must expose an `IndexedTable` class."
    )
    assert hasattr(solution_module, "get_indexed_table"), (
        "solution.py must expose a `get_indexed_table` factory function."
    )


def test_brute_force_search_returns_sentinel(workflow_result):
    sentinel_id = workflow_result["sentinel_id"]
    pre_index_top = workflow_result["pre_index_top"]
    assert int(pre_index_top[0]["id"]) == sentinel_id, (
        f"Pre-index brute-force search must rank sentinel id={sentinel_id} first, "
        f"got id={pre_index_top[0].get('id')}."
    )


def test_final_table_has_at_least_1025_rows(workflow_result):
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    total = db.open_table(VECTORS_TABLE).count_rows()
    assert total >= 1025, (
        f"Verifier should have driven the table past 1024 rows; got {total}."
    )


def test_index_build_log_table_exists(workflow_result):
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    names = db.table_names()
    assert LOG_TABLE in names, (
        f"Expected audit table {LOG_TABLE!r} to exist after threshold crossings; "
        f"found tables: {names}."
    )


def test_index_build_log_has_exactly_three_rows(workflow_result):
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    log = db.open_table(LOG_TABLE)
    n = log.count_rows()
    assert n == EXPECTED_BUILDS, (
        f"Expected exactly {EXPECTED_BUILDS} audit rows (one per threshold crossing), got {n}."
    )


def test_index_build_log_row_counts_match_thresholds(workflow_result):
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    log = db.open_table(LOG_TABLE)
    df = log.to_pandas()
    assert "row_count_at_build" in df.columns, (
        f"Audit table must include a `row_count_at_build` column; got {list(df.columns)}."
    )
    assert "num_partitions" in df.columns, (
        f"Audit table must include a `num_partitions` column; got {list(df.columns)}."
    )
    assert "ts" in df.columns, (
        f"Audit table must include a `ts` column; got {list(df.columns)}."
    )
    counts = sorted(int(v) for v in df["row_count_at_build"].tolist())
    assert len(counts) == EXPECTED_BUILDS, (
        f"Expected {EXPECTED_BUILDS} audit rows after sorting, got {counts}."
    )
    for actual, threshold in zip(counts, THRESHOLDS):
        assert actual >= threshold, (
            f"Audit row with row_count_at_build={actual} should correspond to threshold "
            f">= {threshold}; thresholds were {THRESHOLDS}."
        )
    # num_partitions are positive ints
    for np_value in df["num_partitions"].tolist():
        assert int(np_value) > 0, (
            f"num_partitions must be a positive integer, got {np_value}."
        )
    # ts is non-empty
    for ts in df["ts"].tolist():
        assert isinstance(ts, str) and len(ts) > 0, (
            f"ts must be a non-empty string, got {ts!r}."
        )


def test_ivf_pq_index_present_on_vector_column(workflow_result):
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(VECTORS_TABLE)
    indices = table.list_indices()
    assert len(indices) >= 1, (
        f"Expected at least one index on {VECTORS_TABLE!r}, got {indices}."
    )
    matches = []
    for idx in indices:
        idx_type = str(getattr(idx, "index_type", "")).upper().replace("_", "")
        cols = list(getattr(idx, "columns", []) or [])
        if "IVFPQ" in idx_type and cols == ["vector"]:
            matches.append(idx)
    assert matches, (
        f"Expected an IVF_PQ index on the `vector` column; got {indices}."
    )


def test_post_index_search_returns_sentinel(workflow_result):
    sentinel_id = workflow_result["sentinel_id"]
    post = workflow_result["post_index_top"]
    assert isinstance(post, list) and len(post) == 5, (
        f"search(vec, k=5) after threshold crossings must return a list of length 5, got {post!r}."
    )
    for item in post:
        assert isinstance(item, dict) and "id" in item, (
            f"Each search result must be a dict containing an `id`; got {item!r}."
        )
    top_id = int(post[0]["id"])
    assert top_id == sentinel_id, (
        f"Indexed search must still rank sentinel id={sentinel_id} first, got id={top_id}."
    )
