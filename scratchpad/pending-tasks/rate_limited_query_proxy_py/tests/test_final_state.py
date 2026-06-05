"""Final-state verifier for rate_limited_query_proxy_py.

The verifier interacts with the candidate's `RateLimitedSearchProxy` class
through its public Python API only. A controllable clock is injected so the
time-dependent assertions are deterministic.
"""

import importlib.util
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")


# --- Test infrastructure -----------------------------------------------------


class FakeClock:
    """Manually controllable monotonic clock returning seconds (float)."""

    def __init__(self, start: float = 1000.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID", "")
    assert rid, "ZEALT_RUN_ID must be set during verification"
    return rid


def _bucket_table_name() -> str:
    return f"rate_buckets_{_run_id()}"


def _docs_table_name() -> str:
    return f"documents_{_run_id()}"


@pytest.fixture(scope="module")
def solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"Missing {SOLUTION_PATH}"
    # Make sure the project dir is importable so candidates can use relative imports.
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "RateLimitedSearchProxy"), (
        "solution.py must expose a `RateLimitedSearchProxy` class."
    )
    return module


@pytest.fixture
def fresh_bucket_table():
    """Drop the bucket table before each test so each test starts from a clean state."""
    import lancedb

    db = lancedb.connect(DATA_DIR)
    name = _bucket_table_name()
    if name in db.table_names():
        db.drop_table(name)
    yield
    # After test cleanup is optional; the next test will drop again.


@pytest.fixture
def docs_table():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    return db.open_table(_docs_table_name())


@pytest.fixture
def query_vec():
    rng = np.random.default_rng(7)
    return rng.standard_normal(16).astype(np.float32).tolist()


# --- Tests -------------------------------------------------------------------


def test_import_contract(solution_module, fresh_bucket_table, docs_table, query_vec):
    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )
    res = proxy.search("alice", query_vec, 3)
    assert isinstance(res, dict), "search() must return a dict"
    for key in ("results", "throttled", "retry_after_ms"):
        assert key in res, f"search() response missing '{key}'"
    assert res["throttled"] is False, "First call should not be throttled."
    assert isinstance(res["results"], list) and len(res["results"]) == 3, (
        f"Expected 3 results, got {res['results']!r}"
    )
    first = res["results"][0]
    assert "id" in first, "Each result must contain an 'id' field."
    assert "_distance" in first, "Each result must contain a '_distance' field."


def test_burst_limit_single_user(
    solution_module, fresh_bucket_table, docs_table, query_vec
):
    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )

    # 15 back-to-back calls with the clock frozen (== <100ms wall-clock).
    outcomes = [proxy.search("alice", query_vec, 3) for _ in range(15)]
    successes = [o for o in outcomes if not o["throttled"]]
    throttled = [o for o in outcomes if o["throttled"]]
    assert len(successes) == 10, (
        f"Expected exactly 10 successful calls in a 15-call burst at capacity=10, "
        f"got {len(successes)} (throttled={len(throttled)})."
    )
    assert len(throttled) == 5, (
        f"Expected exactly 5 throttled calls in the burst, got {len(throttled)}."
    )

    for o in successes:
        assert len(o["results"]) == 3, "Each successful call must return 3 results."
        assert o["retry_after_ms"] == 0, (
            "Successful calls should have retry_after_ms == 0, "
            f"got {o['retry_after_ms']}."
        )

    # Retry-after monotonic & bounded.
    first_throttled_retry = throttled[0]["retry_after_ms"]
    assert 0 < first_throttled_retry <= 60000, (
        f"retry_after_ms must be in (0, 60000], got {first_throttled_retry}."
    )
    expected_upper = math.ceil(1000.0 / 5.0) + 5  # 205 ms tolerance
    assert first_throttled_retry <= expected_upper, (
        f"First throttled retry_after_ms ({first_throttled_retry}) exceeds "
        f"expected upper bound ({expected_upper})."
    )


def test_refill_after_wait(solution_module, fresh_bucket_table, docs_table, query_vec):
    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )

    # Drain alice.
    for _ in range(10):
        out = proxy.search("alice", query_vec, 2)
        assert not out["throttled"]
    # Next call must be throttled.
    assert proxy.search("alice", query_vec, 2)["throttled"] is True

    # Advance clock by 0.6s -> 3 refilled tokens.
    clock.advance(0.6)

    refilled = [proxy.search("alice", query_vec, 2) for _ in range(3)]
    for i, o in enumerate(refilled):
        assert not o["throttled"], (
            f"Refilled call #{i + 1} should succeed after 0.6 s, but was throttled."
        )

    # 4th immediate call must be throttled (only 3 tokens refilled).
    next_o = proxy.search("alice", query_vec, 2)
    assert next_o["throttled"] is True, (
        "4th post-wait call should be throttled (only 3 tokens refilled)."
    )


def test_per_user_isolation(
    solution_module, fresh_bucket_table, docs_table, query_vec
):
    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )

    # Drain alice.
    for _ in range(10):
        assert not proxy.search("alice", query_vec, 2)["throttled"]
    assert proxy.search("alice", query_vec, 2)["throttled"] is True, (
        "alice should be throttled after draining her bucket."
    )

    # bob starts with a fresh full bucket.
    bob_out = proxy.search("bob", query_vec, 2)
    assert not bob_out["throttled"], (
        "bob should succeed because per_user buckets are independent."
    )


def test_parallel_two_users(
    solution_module, fresh_bucket_table, docs_table, query_vec
):
    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )

    def call(user_id):
        return proxy.search(user_id, query_vec, 2)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = []
        for _ in range(5):
            futures.append(ex.submit(call, "alice"))
            futures.append(ex.submit(call, "bob"))
        results = [f.result() for f in futures]

    alice_results = [r for r, uid in zip(results, ["alice", "bob"] * 5) if uid == "alice"]
    bob_results = [r for r, uid in zip(results, ["alice", "bob"] * 5) if uid == "bob"]

    alice_ok = sum(1 for r in alice_results if not r["throttled"])
    bob_ok = sum(1 for r in bob_results if not r["throttled"])
    assert alice_ok == 5, f"alice should have 5 successes in parallel run, got {alice_ok}"
    assert bob_ok == 5, f"bob should have 5 successes in parallel run, got {bob_ok}"


def test_persistence_after_restart(
    solution_module, fresh_bucket_table, docs_table, query_vec
):
    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )

    # Consume 7 tokens for alice -> 3 remaining.
    for _ in range(7):
        out = proxy.search("alice", query_vec, 2)
        assert not out["throttled"]

    # Destroy proxy.
    del proxy

    # Recreate with the same frozen clock (no time has passed).
    new_proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )

    # 3 calls back-to-back should succeed, then the 4th should be throttled
    # because the bucket only had 3 tokens left after restart.
    outcomes = [new_proxy.search("alice", query_vec, 2) for _ in range(4)]
    successes = sum(1 for o in outcomes if not o["throttled"])
    throttled = sum(1 for o in outcomes if o["throttled"])
    assert successes == 3 and throttled == 1, (
        "After restart, the recovered bucket should have ~3 tokens. "
        f"Got successes={successes}, throttled={throttled}. "
        "If 4 succeeded, the proxy is refilling to full on restart instead of persisting."
    )


def test_bucket_table_schema_and_rows(
    solution_module, fresh_bucket_table, docs_table, query_vec
):
    import lancedb

    clock = FakeClock()
    proxy = solution_module.RateLimitedSearchProxy(
        docs_table, capacity=10, refill_per_sec=5.0, per_user=True, clock=clock
    )
    # Touch buckets for both users.
    proxy.search("alice", query_vec, 2)
    proxy.search("bob", query_vec, 2)

    db = lancedb.connect(DATA_DIR)
    name = _bucket_table_name()
    assert name in db.table_names(), (
        f"Expected bucket table {name} to be created."
    )
    tbl = db.open_table(name)
    schema = tbl.schema
    names = set(schema.names)
    required = {"user_id", "tokens", "last_refill_ts"}
    assert required.issubset(names), (
        f"Bucket table schema must contain {required}, got {names}."
    )

    # Type checks (LanceDB uses pyarrow types).
    field_types = {f.name: str(f.type) for f in schema}
    assert "string" in field_types["user_id"].lower() or "utf8" in field_types["user_id"].lower(), (
        f"user_id must be utf8/string, got {field_types['user_id']}"
    )
    assert "double" in field_types["tokens"].lower() or "float64" in field_types["tokens"].lower(), (
        f"tokens must be float64, got {field_types['tokens']}"
    )
    assert "int64" in field_types["last_refill_ts"].lower(), (
        f"last_refill_ts must be int64, got {field_types['last_refill_ts']}"
    )

    df = tbl.to_pandas()
    users = set(df["user_id"].tolist())
    assert {"alice", "bob"}.issubset(users), (
        f"Expected both alice and bob in bucket table, got {users}."
    )
