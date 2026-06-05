import importlib
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import redis

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/home/user/myproject/lancedb_data"
REDIS_URL = "redis://localhost:6379/0"


@pytest.fixture(scope="session")
def run_id():
    val = os.environ.get("ZEALT_RUN_ID")
    assert val, "ZEALT_RUN_ID environment variable is not set."
    return val


@pytest.fixture(scope="session")
def solution_module():
    sys.path.insert(0, PROJECT_DIR)
    mod = importlib.import_module("solution")
    return mod


@pytest.fixture(scope="session")
def table(run_id):
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(f"docs_{run_id}")
    return tbl


@pytest.fixture(scope="session")
def query_vector(run_id):
    rng = np.random.default_rng(20260605)
    return rng.standard_normal(32).astype(np.float32)


@pytest.fixture
def redis_client():
    return redis.Redis.from_url(REDIS_URL, decode_responses=False)


def _clean(cs, redis_client):
    """Reset cache state between scenarios."""
    cs.invalidate_table()
    # Defensive belt-and-braces: also flush DB to remove unrelated keys
    redis_client.flushdb()


def test_redis_ping():
    res = subprocess.run(
        ["redis-cli", "-h", "127.0.0.1", "-p", "6379", "PING"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"redis-cli PING failed: {res.stderr}"
    assert res.stdout.strip() == "PONG", f"Expected PONG, got {res.stdout!r}"


def test_constructor_signature(solution_module, table):
    cs = solution_module.CachedSearch(table, REDIS_URL, ttl_seconds=60)
    assert hasattr(cs, "search"), "CachedSearch has no search() method."
    assert hasattr(cs, "invalidate_table"), "CachedSearch has no invalidate_table() method."
    assert callable(cs.search) and callable(cs.invalidate_table)


def test_cold_miss_then_warm_hit_with_speedup(
    solution_module, table, query_vector, redis_client
):
    cs = solution_module.CachedSearch(table, REDIS_URL, ttl_seconds=60)
    _clean(cs, redis_client)

    first = cs.search(query_vector, k=5)
    assert isinstance(first, dict), f"search() must return dict, got {type(first)}"
    for key in ("results", "cache_hit", "latency_ms"):
        assert key in first, f"search() result missing key {key!r}: {first.keys()}"
    assert first["cache_hit"] is False, f"First call should miss, got cache_hit={first['cache_hit']}"
    assert isinstance(first["results"], list), "results must be a list"
    assert len(first["results"]) == 5, f"Expected 5 results, got {len(first['results'])}"
    assert isinstance(first["latency_ms"], (int, float)), "latency_ms must be numeric"

    second = cs.search(query_vector, k=5)
    assert second["cache_hit"] is True, (
        f"Second identical call should hit, got cache_hit={second['cache_hit']}"
    )
    assert len(second["results"]) == 5, f"Expected 5 results on hit, got {len(second['results'])}"

    # Latency check: hit should be at least 5x faster than miss.
    assert first["latency_ms"] > 0, "First latency must be positive."
    assert second["latency_ms"] > 0, "Second latency must be positive."
    ratio = first["latency_ms"] / max(second["latency_ms"], 1e-6)
    assert ratio >= 5.0, (
        f"Cache hit not at least 5x faster than miss: "
        f"first={first['latency_ms']:.3f}ms, second={second['latency_ms']:.3f}ms, ratio={ratio:.2f}"
    )

    # Result payload identity check on a stable subset
    def _ids(results):
        return [r.get("id") for r in results]

    assert _ids(first["results"]) == _ids(second["results"]), (
        "Cached results' ids must match the cold-miss ids."
    )


def test_independent_cache_entry_per_k(
    solution_module, table, query_vector, redis_client
):
    cs = solution_module.CachedSearch(table, REDIS_URL, ttl_seconds=60)
    _clean(cs, redis_client)

    first = cs.search(query_vector, k=5)
    assert first["cache_hit"] is False
    second = cs.search(query_vector, k=5)
    assert second["cache_hit"] is True

    # Now hit it with a different k
    third = cs.search(query_vector, k=10)
    assert third["cache_hit"] is False, (
        f"k=10 with same query must miss, got cache_hit={third['cache_hit']}"
    )
    assert len(third["results"]) == 10, f"Expected 10 results for k=10, got {len(third['results'])}"
    fourth = cs.search(query_vector, k=10)
    assert fourth["cache_hit"] is True, "Second k=10 call must hit."


def test_invalidate_table_clears_entries(
    solution_module, table, query_vector, redis_client, run_id
):
    cs = solution_module.CachedSearch(table, REDIS_URL, ttl_seconds=60)
    _clean(cs, redis_client)

    cs.search(query_vector, k=5)
    warm = cs.search(query_vector, k=5)
    assert warm["cache_hit"] is True, "Pre-invalidate warm call should hit."

    # Confirm at least one key exists with the table prefix
    table_name = f"docs_{run_id}"
    matching_keys_before = [
        k for k in redis_client.scan_iter(match=f"*{table_name}*", count=1000)
    ]
    assert matching_keys_before, (
        f"Expected at least one Redis key containing table name {table_name!r} before invalidate."
    )

    cs.invalidate_table()

    matching_keys_after = [
        k for k in redis_client.scan_iter(match=f"*{table_name}*", count=1000)
    ]
    assert matching_keys_after == [], (
        f"invalidate_table() must remove all keys for {table_name}, "
        f"still present: {matching_keys_after}"
    )

    miss_again = cs.search(query_vector, k=5)
    assert miss_again["cache_hit"] is False, (
        "After invalidate_table(), the next identical search must miss."
    )


def test_ttl_expiry(solution_module, table, query_vector, redis_client):
    cs = solution_module.CachedSearch(table, REDIS_URL, ttl_seconds=2)
    _clean(cs, redis_client)

    cs.search(query_vector, k=3)
    warm = cs.search(query_vector, k=3)
    assert warm["cache_hit"] is True, "Pre-expiry call should hit."

    time.sleep(3)

    after = cs.search(query_vector, k=3)
    assert after["cache_hit"] is False, (
        f"After ttl_seconds+1 sleep, entry must be expired and the search must miss; "
        f"got cache_hit={after['cache_hit']}"
    )


def test_text_query_round_trip(solution_module, table, redis_client):
    cs = solution_module.CachedSearch(table, REDIS_URL, ttl_seconds=60)
    _clean(cs, redis_client)

    miss = cs.search("first call text", k=4)
    assert miss["cache_hit"] is False, "First text call should miss."
    hit = cs.search("first call text", k=4)
    assert hit["cache_hit"] is True, "Second identical text call should hit."

    def _ids(results):
        return [r.get("id") for r in results]

    assert _ids(miss["results"]) == _ids(hit["results"]), (
        "Cached text results must match the cold-miss results."
    )
