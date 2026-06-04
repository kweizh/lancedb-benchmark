import os
import shutil
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
DB_PATH = os.path.join(PROJECT_DIR, "data.lancedb")
ARTICLES_TABLE = "articles"
LOGS_TABLE = "query_logs"

CALLS = [
    {"query_id": "q-001", "user_id": "u-alice", "query_text": "deep learning"},
    {"query_id": "q-002", "user_id": "u-bob",   "query_text": "vector databases"},
    {"query_id": "q-003", "user_id": "u-alice", "query_text": "retrieval augmented generation"},
    {"query_id": "q-004", "user_id": "u-carol", "query_text": ""},
    {"query_id": "q-005", "user_id": "u-bob",   "query_text": "audit logging"},
]
TOP_K = 5
N = len(CALLS)


def _remove_logs_table_dir():
    """Make sure the query_logs table does not exist before exercising the candidate code."""
    for candidate in (
        os.path.join(DB_PATH, f"{LOGS_TABLE}.lance"),
        os.path.join(DB_PATH, LOGS_TABLE),
    ):
        if os.path.isdir(candidate):
            shutil.rmtree(candidate)
        elif os.path.isfile(candidate):
            os.remove(candidate)


@pytest.fixture(scope="module")
def loaded_solution():
    sys.path.insert(0, PROJECT_DIR)
    try:
        from solution import LoggedSearcher  # noqa: WPS433  (intentional dynamic import)
    except Exception as exc:  # pragma: no cover - surfaced via pytest failure
        pytest.fail(f"Failed to import LoggedSearcher from {PROJECT_DIR}/solution.py: {exc!r}")
    return LoggedSearcher


@pytest.fixture(scope="module")
def run_results(loaded_solution):
    """Run the N searches with the candidate's LoggedSearcher and capture results."""
    _remove_logs_table_dir()

    LoggedSearcher = loaded_solution
    searcher = LoggedSearcher(
        db_uri=DB_PATH,
        articles_table=ARTICLES_TABLE,
        logs_table=LOGS_TABLE,
    )

    rng = np.random.default_rng(seed=2026)
    query_vectors = rng.standard_normal((N, 64)).astype(np.float32)

    returned_hits = []
    for i, meta in enumerate(CALLS):
        qv = query_vectors[i]
        # Pass numpy array directly to mirror normal LanceDB usage.
        hits = searcher.search(
            qv,
            TOP_K,
            query_id=meta["query_id"],
            user_id=meta["user_id"],
            query_text=meta["query_text"],
        )
        returned_hits.append(hits)
        # Sleep a tiny amount to keep timestamps strictly observable as non-decreasing.
        time.sleep(0.01)

    return {
        "searcher": searcher,
        "query_vectors": query_vectors,
        "returned_hits": returned_hits,
    }


def test_solution_module_importable():
    """The candidate must expose a LoggedSearcher class."""
    sys.path.insert(0, PROJECT_DIR)
    import solution

    assert hasattr(solution, "LoggedSearcher"), "solution.py must export a class named 'LoggedSearcher'."


def test_search_returns_list_with_required_keys(run_results):
    for i, hits in enumerate(run_results["returned_hits"]):
        assert isinstance(hits, list), f"Call {i}: searcher.search must return a list, got {type(hits)!r}."
        assert len(hits) == TOP_K, f"Call {i}: expected {TOP_K} hits, got {len(hits)}."
        for j, h in enumerate(hits):
            # Each hit must include at least 'id' and 'title'
            assert "id" in h, f"Call {i} hit {j}: missing required key 'id'. Got keys: {list(h.keys())}"
            assert "title" in h, f"Call {i} hit {j}: missing required key 'title'. Got keys: {list(h.keys())}"


def test_query_logs_table_exists():
    db = lancedb.connect(DB_PATH)
    names = db.table_names()
    assert LOGS_TABLE in names, (
        f"Expected '{LOGS_TABLE}' table to be created lazily by LoggedSearcher. "
        f"Found tables: {names}"
    )


def test_query_logs_row_count(run_results):
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(LOGS_TABLE)
    n = tbl.count_rows()
    assert n == N, f"Expected {N} audit rows in '{LOGS_TABLE}', found {n}."


def _load_logs_in_insertion_order():
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(LOGS_TABLE)
    df = tbl.to_pandas()
    # Sort by ts to obtain the audit order. ts must be monotonic non-decreasing,
    # so this sort is equivalent to insertion order if the contract is honored.
    if "ts" in df.columns:
        df_sorted = df.sort_values("ts", kind="stable").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)
    return df, df_sorted


def test_logs_have_required_columns():
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(LOGS_TABLE)
    schema_names = {f.name for f in tbl.schema}
    required = {"query_id", "user_id", "query_text", "ts", "latency_ms", "hit_count", "top_ids"}
    missing = required - schema_names
    assert not missing, (
        f"query_logs table is missing required columns {missing}. Found columns: {schema_names}."
    )


def test_timestamps_monotonic_non_decreasing():
    df, _ = _load_logs_in_insertion_order()
    # Map by query_id to recover physical issue order
    order_by_qid = {meta["query_id"]: i for i, meta in enumerate(CALLS)}
    df_ordered = df.copy()
    df_ordered["__order"] = df_ordered["query_id"].map(order_by_qid)
    assert df_ordered["__order"].notna().all(), (
        "query_logs contains rows whose query_id was not one of the issued query ids."
    )
    df_ordered = df_ordered.sort_values("__order").reset_index(drop=True)

    ts_series = df_ordered["ts"]
    ts_ns = pa.array(ts_series.values).cast(pa.timestamp("ns")).to_pylist()
    # Convert to integer ns
    ts_ints = []
    for v in ts_ns:
        # v is datetime or pd.Timestamp
        if hasattr(v, "value"):
            ts_ints.append(int(v.value))
        else:
            # datetime -> convert
            import datetime as _dt
            assert isinstance(v, _dt.datetime), f"Unexpected ts type: {type(v)!r}"
            ts_ints.append(int(v.timestamp() * 1_000_000_000))

    for i in range(1, len(ts_ints)):
        assert ts_ints[i] >= ts_ints[i - 1], (
            f"Timestamps must be monotonically non-decreasing in insertion order. "
            f"ts[{i-1}]={ts_ints[i-1]} ts[{i}]={ts_ints[i]}"
        )


def test_latency_ms_strictly_positive():
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(LOGS_TABLE)
    df = tbl.to_pandas()
    for i, val in enumerate(df["latency_ms"].tolist()):
        assert val is not None, f"Row {i}: latency_ms is null."
        assert float(val) > 0.0, f"Row {i}: expected latency_ms > 0, got {val}."


def _coerce_id_list(value):
    """Coerce a row's top_ids field into a flat Python list[int]."""
    if value is None:
        return None
    # numpy array / list / pa.ListScalar / similar
    try:
        out = list(value)
    except TypeError:
        return None
    return [int(x) for x in out]


def test_top_ids_match_raw_search(run_results):
    db = lancedb.connect(DB_PATH)
    articles_tbl = db.open_table(ARTICLES_TABLE)
    logs_tbl = db.open_table(LOGS_TABLE)
    df = logs_tbl.to_pandas()

    # Index by query_id for matching
    rows_by_qid = {row["query_id"]: row for _, row in df.iterrows()}

    qvs = run_results["query_vectors"]
    for i, meta in enumerate(CALLS):
        qid = meta["query_id"]
        assert qid in rows_by_qid, f"Audit log is missing row for query_id={qid}."
        row = rows_by_qid[qid]

        # Independent ground-truth using raw search
        truth_hits = articles_tbl.search(qvs[i]).limit(TOP_K).to_list()
        truth_ids = [int(h["id"]) for h in truth_hits]

        logged_ids = _coerce_id_list(row["top_ids"])
        assert logged_ids is not None, f"Row for {qid}: 'top_ids' could not be coerced to a list."
        assert logged_ids == truth_ids, (
            f"Row for {qid}: top_ids mismatch. Expected {truth_ids}, got {logged_ids}."
        )

        # Cross-check against what LoggedSearcher actually returned for this call.
        returned_ids = [int(h["id"]) for h in run_results["returned_hits"][i]]
        assert returned_ids == truth_ids, (
            f"LoggedSearcher.search returned ids {returned_ids} but raw search produced {truth_ids} "
            f"for call {qid}; LoggedSearcher must return the same hits as the underlying table.search()."
        )


def test_hit_count_consistency():
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(LOGS_TABLE)
    df = tbl.to_pandas()
    for _, row in df.iterrows():
        hc = int(row["hit_count"])
        top_ids = _coerce_id_list(row["top_ids"])
        assert top_ids is not None, f"Row {row.get('query_id')}: top_ids is missing/not a list."
        assert hc == len(top_ids), (
            f"Row {row.get('query_id')}: hit_count={hc} does not match len(top_ids)={len(top_ids)}."
        )
        assert hc == TOP_K, f"Row {row.get('query_id')}: expected hit_count={TOP_K}, got {hc}."


def test_metadata_round_trip():
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(LOGS_TABLE)
    df = tbl.to_pandas()
    rows_by_qid = {row["query_id"]: row for _, row in df.iterrows()}

    for meta in CALLS:
        qid = meta["query_id"]
        assert qid in rows_by_qid, f"Audit log is missing row for query_id={qid}."
        row = rows_by_qid[qid]
        assert str(row["user_id"]) == meta["user_id"], (
            f"Row {qid}: expected user_id={meta['user_id']!r}, got {row['user_id']!r}."
        )
        assert str(row["query_text"]) == meta["query_text"], (
            f"Row {qid}: expected query_text={meta['query_text']!r}, got {row['query_text']!r}."
        )


def test_search_fidelity_recall(run_results):
    """Independently re-run one search and confirm LoggedSearcher matches the raw API."""
    searcher = run_results["searcher"]
    qv = run_results["query_vectors"][2]  # call 3

    logged = searcher.search(
        qv,
        TOP_K,
        query_id="q-recheck",
        user_id="u-recheck",
        query_text="recheck",
    )
    logged_ids = [int(h["id"]) for h in logged]

    db = lancedb.connect(DB_PATH)
    raw = db.open_table(ARTICLES_TABLE).search(qv).limit(TOP_K).to_list()
    raw_ids = [int(h["id"]) for h in raw]

    assert logged_ids == raw_ids, (
        f"LoggedSearcher.search must return the same id ordering as the underlying table.search(). "
        f"Got {logged_ids}, expected {raw_ids}."
    )
