import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import lancedb
import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")


# ---------------------------------------------------------------------------
# Deterministic write plan: 4 threads x 25 batches x 10 ids = 1000 ops.
# Covers every id in 0..199. Per-id expected final ts = max ts that wrote it.
# ---------------------------------------------------------------------------
def _build_plan():
    plan = []  # list of (thread_id, batch_id, ts, rows)
    expected_max_ts = {i: 0 for i in range(200)}  # initial seeded ts is 0
    for t in range(4):
        for b in range(25):
            batch_id = f"t{t}-b{b}"
            ts = t * 1000 + b + 1  # strictly positive, unique
            ids = sorted({((t * 25 * 10) + (b * 10) + j) % 200 for j in range(10)})
            rows = [{"id": int(i), "value": int(ts), "ts": int(ts)} for i in ids]
            for i in ids:
                if ts > expected_max_ts[i]:
                    expected_max_ts[i] = ts
            plan.append((t, batch_id, ts, rows))
    return plan, expected_max_ts


def _load_safe_writer():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    assert spec is not None, f"Cannot locate solution module at {SOLUTION_PATH}."
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "SafeWriter"), "solution.py must define class SafeWriter."
    return mod.SafeWriter


@pytest.fixture(scope="module")
def env_config():
    uri = os.environ.get("LANCEDB_URI")
    name = os.environ.get("LANCEDB_TABLE")
    assert uri, "LANCEDB_URI must be set in the environment."
    assert name, "LANCEDB_TABLE must be set in the environment."
    return uri, name


@pytest.fixture(scope="module")
def clean_database(env_config):
    """Remove any pre-existing write_attempts table so we observe this run only."""
    uri, _ = env_config
    db = lancedb.connect(uri)
    if "write_attempts" in db.table_names():
        db.drop_table("write_attempts")
    return uri


@pytest.fixture(scope="module")
def concurrent_run(env_config, clean_database):
    """Run the candidate's SafeWriter concurrently and yield the plan + expected max ts."""
    uri, name = env_config
    SafeWriter = _load_safe_writer()
    plan, expected_max_ts = _build_plan()

    # Group plan by thread id
    per_thread = {t: [] for t in range(4)}
    for t, batch_id, ts, rows in plan:
        per_thread[t].append((batch_id, ts, rows))

    def worker(thread_id):
        # Each thread builds its own SafeWriter so connection state is independent.
        writer = SafeWriter(db_uri=uri, table_name=name, key="id")
        for batch_id, _ts, rows in per_thread[thread_id]:
            writer.upsert(batch_id, rows)
        return thread_id

    errors = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(worker, t): t for t in range(4)}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # pragma: no cover
                errors.append((futs[fut], repr(e)))

    assert not errors, f"SafeWriter threads raised unhandled errors: {errors!r}"
    return uri, name, plan, expected_max_ts


def test_solution_file_exists():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Expected candidate solution module at {SOLUTION_PATH}."
    )


def test_safewriter_constructor_signature():
    SafeWriter = _load_safe_writer()
    uri = os.environ["LANCEDB_URI"]
    name = os.environ["LANCEDB_TABLE"]
    sw = SafeWriter(db_uri=uri, table_name=name, key="id")
    assert hasattr(sw, "upsert"), "SafeWriter must expose an upsert(batch_id, rows) method."


def test_final_row_count_is_200(concurrent_run):
    uri, name, _, _ = concurrent_run
    db = lancedb.connect(uri)
    tbl = db.open_table(name)
    n = tbl.count_rows()
    assert n == 200, (
        f"Target table must contain exactly 200 rows after the concurrent run; got {n}."
    )


def test_final_ids_cover_full_space(concurrent_run):
    uri, name, _, _ = concurrent_run
    db = lancedb.connect(uri)
    tbl = db.open_table(name)
    df = tbl.to_pandas()
    ids = set(int(x) for x in df["id"].tolist())
    expected = set(range(200))
    missing = expected - ids
    extra = ids - expected
    assert not missing and not extra, (
        f"id set must be {{0..199}}; missing={sorted(missing)} extra={sorted(extra)}"
    )


def test_final_values_equal_expected_max_ts(concurrent_run):
    uri, name, _, expected_max_ts = concurrent_run
    db = lancedb.connect(uri)
    tbl = db.open_table(name)
    df = tbl.to_pandas().sort_values("id").reset_index(drop=True)
    mismatches = []
    for _, row in df.iterrows():
        i = int(row["id"])
        actual_value = int(row["value"])
        actual_ts = int(row["ts"])
        expected = expected_max_ts[i]
        if actual_value != expected or actual_ts != expected:
            mismatches.append((i, expected, actual_value, actual_ts))
    assert not mismatches, (
        "For each id, final value and ts must equal the largest ts of any batch "
        f"that wrote to it. First few mismatches (id, expected, got_value, got_ts): "
        f"{mismatches[:10]!r}"
    )


def test_write_attempts_table_exists(concurrent_run):
    uri, _, _, _ = concurrent_run
    db = lancedb.connect(uri)
    assert "write_attempts" in db.table_names(), (
        "SafeWriter must create a 'write_attempts' table on first use."
    )


def test_write_attempts_schema_and_per_batch_success(concurrent_run):
    uri, _, plan, _ = concurrent_run
    db = lancedb.connect(uri)
    att = db.open_table("write_attempts").to_pandas()
    required_cols = {"batch_id", "attempt_num", "success", "error_msg", "ts"}
    missing_cols = required_cols - set(att.columns)
    assert not missing_cols, (
        f"write_attempts table missing required columns: {sorted(missing_cols)}; "
        f"found {sorted(att.columns)}"
    )

    expected_batch_ids = {batch_id for _, batch_id, _, _ in plan}
    actual_batch_ids = set(str(x) for x in att["batch_id"].tolist())
    missing_batches = expected_batch_ids - actual_batch_ids
    assert not missing_batches, (
        f"write_attempts is missing log rows for {len(missing_batches)} batch_ids; "
        f"first few missing: {sorted(missing_batches)[:5]!r}"
    )

    # Every batch must have at least one success=True row.
    failed_to_complete = []
    for batch_id in expected_batch_ids:
        sub = att[att["batch_id"].astype(str) == batch_id]
        if not sub["success"].any():
            failed_to_complete.append(batch_id)
    assert not failed_to_complete, (
        "Every batch must have at least one successful attempt logged; "
        f"batches with no success row: {failed_to_complete[:5]!r}"
    )


def test_at_least_one_retry_occurred(concurrent_run):
    uri, _, _, _ = concurrent_run
    db = lancedb.connect(uri)
    att = db.open_table("write_attempts").to_pandas()

    # Look for any batch_id with both a failed attempt and a later successful attempt.
    retried_batches = []
    for batch_id, sub in att.groupby(att["batch_id"].astype(str)):
        had_failure = (~sub["success"].astype(bool)).any()
        had_success = sub["success"].astype(bool).any()
        if had_failure and had_success:
            retried_batches.append(batch_id)

    # Also accept attempt_num > 0 with success=True as evidence of a retry.
    max_attempt = int(att["attempt_num"].max()) if len(att) else -1

    assert retried_batches or max_attempt > 0, (
        "Expected at least one batch to record a failed attempt followed by a "
        "successful retry (or any attempt_num > 0). The wrapper should detect "
        "contention and recover. attempts table size="
        f"{len(att)}, max attempt_num={max_attempt}."
    )


def test_attempt_num_starts_at_zero(concurrent_run):
    uri, _, _, _ = concurrent_run
    db = lancedb.connect(uri)
    att = db.open_table("write_attempts").to_pandas()
    min_attempt = int(att["attempt_num"].min())
    assert min_attempt == 0, (
        f"Expected the first attempt for at least one batch to be attempt_num=0; "
        f"got min attempt_num={min_attempt}."
    )
