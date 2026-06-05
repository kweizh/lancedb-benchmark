import importlib
import os
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")

BASE_TS = 1704067200  # 2024-01-01T00:00:00Z, unix seconds


def _run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set for the verifier."
    return rid


def _table_name():
    return f"events_{_run_id()}"


@pytest.fixture(scope="module")
def solution_module():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Expected solution module at {SOLUTION_PATH}"
    )
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    mod = importlib.import_module("solution")
    return mod


@pytest.fixture(scope="module")
def snapshot():
    """Load the seeded events table into a numpy/pandas-friendly dict for
    brute-force ground-truth computation."""
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(_table_name())
    df = tbl.to_pandas()
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"

    ids = df["id"].to_numpy().astype(np.int64)
    timestamps = df["timestamp"].to_numpy().astype(np.int64)
    event_types = df["event_type"].astype(str).to_numpy()
    payloads = df["payload"].astype(str).to_numpy()
    vectors = np.stack([np.asarray(v, dtype=np.float32) for v in df["vector"].tolist()])
    assert vectors.shape == (1000, 32), vectors.shape

    order = np.argsort(ids)
    return {
        "id": ids[order],
        "timestamp": timestamps[order],
        "event_type": event_types[order],
        "payload": payloads[order],
        "vector": vectors[order],
    }


def _brute_force_topk(snapshot, mask, query_vec, k):
    ids = snapshot["id"][mask]
    vecs = snapshot["vector"][mask]
    if ids.size == 0:
        return []
    diffs = vecs - query_vec.astype(np.float32)
    dist = np.sqrt(np.sum(diffs * diffs, axis=1))
    # tie-break by id ASC -> use lexsort with primary key = distance
    sort_idx = np.lexsort((ids, dist))
    take = sort_idx[: min(k, ids.size)]
    return ids[take].tolist()


def _rand_query(seed):
    return np.random.default_rng(seed).standard_normal(32).astype(np.float32)


def test_solution_module_exists():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Expected solution module at {SOLUTION_PATH}"
    )


def test_solution_exposes_three_callables(solution_module):
    for name in ("window_search", "bucketed_search", "top_k_per_event_type"):
        fn = getattr(solution_module, name, None)
        assert callable(fn), (
            f"solution.py is missing required top-level callable {name!r}."
        )


# ---------- window_search ----------

# (start_ts, end_ts) windows in unix seconds, k
WINDOW_CASES = [
    # 2024-04-01 -> 2024-07-01
    (1711929600, 1719792000, 10),
    # 2025-01-01 -> 2025-06-15
    (1735689600, 1749945600, 10),
    # 2025-09-01 -> 2026-01-01
    (1756684800, 1767225600, 10),
]


@pytest.mark.parametrize("seed", [11, 22])
@pytest.mark.parametrize("start_ts,end_ts,k", WINDOW_CASES)
def test_window_search_exact_order(solution_module, snapshot, seed, start_ts, end_ts, k):
    q = _rand_query(seed)
    mask = (snapshot["timestamp"] >= start_ts) & (snapshot["timestamp"] < end_ts)
    expected_ids = _brute_force_topk(snapshot, mask, q, k)

    result = solution_module.window_search(q.tolist(), start_ts, end_ts, k)
    assert isinstance(result, list), (
        f"window_search must return a list; got {type(result).__name__}"
    )
    assert len(result) == min(k, int(mask.sum())), (
        f"window_search returned {len(result)} rows; expected {min(k, int(mask.sum()))}."
    )
    for row in result:
        for required in ("id", "timestamp", "event_type", "payload"):
            assert required in row, (
                f"Result row missing required key {required!r}: {row}"
            )
    actual_ids = [int(row["id"]) for row in result]
    assert actual_ids == expected_ids, (
        f"window_search id order mismatch for seed={seed} window=({start_ts},{end_ts}).\n"
        f"Expected: {expected_ids}\n"
        f"Actual:   {actual_ids}"
    )
    # All timestamps must be strictly within the window.
    for row in result:
        ts = int(row["timestamp"])
        assert start_ts <= ts < end_ts, (
            f"window_search returned row with timestamp {ts} outside [{start_ts}, {end_ts})"
        )


def test_window_search_determinism(solution_module):
    q = _rand_query(11)
    start_ts, end_ts, k = WINDOW_CASES[0]
    a = solution_module.window_search(q.tolist(), start_ts, end_ts, k)
    b = solution_module.window_search(q.tolist(), start_ts, end_ts, k)
    assert [int(r["id"]) for r in a] == [int(r["id"]) for r in b], (
        "window_search must be deterministic across identical calls."
    )


# ---------- bucketed_search ----------

BUCKET_CASES = [
    # (bucket_seconds, num_buckets, k_per_bucket, seed)
    (90 * 86400, 8, 5, 33),
    (30 * 86400, 24, 3, 44),
]


@pytest.mark.parametrize("bucket_seconds,num_buckets,k_per_bucket,seed", BUCKET_CASES)
def test_bucketed_search_exact_order(solution_module, snapshot, bucket_seconds, num_buckets, k_per_bucket, seed):
    q = _rand_query(seed)
    result = solution_module.bucketed_search(
        q.tolist(), bucket_seconds, num_buckets, k_per_bucket
    )
    assert isinstance(result, dict), (
        f"bucketed_search must return a dict; got {type(result).__name__}"
    )
    expected_keys = {BASE_TS + i * bucket_seconds for i in range(num_buckets)}
    actual_keys = set(int(k) for k in result.keys())
    assert actual_keys == expected_keys, (
        f"bucketed_search key-set mismatch.\nExpected: {sorted(expected_keys)}\nActual:   {sorted(actual_keys)}"
    )

    for i in range(num_buckets):
        start = BASE_TS + i * bucket_seconds
        end = start + bucket_seconds
        mask = (snapshot["timestamp"] >= start) & (snapshot["timestamp"] < end)
        expected_ids = _brute_force_topk(snapshot, mask, q, k_per_bucket)

        bucket = result.get(start)
        if bucket is None:
            bucket = result.get(int(start))
        assert bucket is not None, f"Missing bucket key {start} in bucketed_search output"
        assert isinstance(bucket, list), (
            f"Bucket value for key {start} must be a list; got {type(bucket).__name__}"
        )
        actual_ids = [int(row["id"]) for row in bucket]
        assert actual_ids == expected_ids, (
            f"bucketed_search id order mismatch for bucket {i} (start={start}).\n"
            f"Expected: {expected_ids}\n"
            f"Actual:   {actual_ids}"
        )
        for row in bucket:
            ts = int(row["timestamp"])
            assert start <= ts < end, (
                f"bucketed_search row in bucket {i} has timestamp {ts} outside [{start}, {end})"
            )


# ---------- top_k_per_event_type ----------


@pytest.mark.parametrize("seed,k_per_type", [(55, 7), (66, 3)])
def test_top_k_per_event_type_exact_order(solution_module, snapshot, seed, k_per_type):
    q = _rand_query(seed)
    distinct_types = sorted(set(snapshot["event_type"].tolist()))

    result = solution_module.top_k_per_event_type(q.tolist(), k_per_type)
    assert isinstance(result, dict), (
        f"top_k_per_event_type must return a dict; got {type(result).__name__}"
    )
    actual_types = sorted(map(str, result.keys()))
    assert actual_types == distinct_types, (
        f"top_k_per_event_type key-set mismatch.\nExpected: {distinct_types}\nActual:   {actual_types}"
    )

    for et in distinct_types:
        mask = snapshot["event_type"] == et
        expected_ids = _brute_force_topk(snapshot, mask, q, k_per_type)
        bucket = result[et]
        assert isinstance(bucket, list), (
            f"Value for event_type {et!r} must be a list; got {type(bucket).__name__}"
        )
        actual_ids = [int(row["id"]) for row in bucket]
        assert actual_ids == expected_ids, (
            f"top_k_per_event_type id order mismatch for event_type={et!r}.\n"
            f"Expected: {expected_ids}\n"
            f"Actual:   {actual_ids}"
        )
        for row in bucket:
            assert row["event_type"] == et, (
                f"Row {row} returned under event_type {et!r} actually has event_type {row.get('event_type')!r}"
            )


# ---------- table not mutated ----------


def test_events_table_not_mutated(snapshot):
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(_table_name())
    assert tbl.count_rows() == 1000, (
        f"Events table row count changed; expected 1000, got {tbl.count_rows()}"
    )
