import importlib
import json
import os
import sys
import time

import numpy as np
import pytest
from pymemcache.client.base import Client as MemcacheClient

PROJECT_DIR = "/home/user/myproject"
FIXTURE_PATH = os.path.join(PROJECT_DIR, "fixture.json")
MEMCACHED_HOST = "127.0.0.1"
MEMCACHED_PORT = 11211

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _load_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


FIXTURE = _load_fixture()
DB_PATH = FIXTURE["db_path"]
TABLE_NAME = FIXTURE["table_name"]
DIM = int(FIXTURE["dim"])


@pytest.fixture(scope="session")
def solution_module():
    import solution  # noqa

    return importlib.reload(solution)


@pytest.fixture(scope="session")
def search_counter():
    """Patch LanceDB's Table.search so we can tell hits (no call) from misses (call)."""
    import lancedb

    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    table_class = type(tbl)
    original_search = table_class.search
    counter = {"n": 0}

    def counting_search(self, *args, **kwargs):
        counter["n"] += 1
        return original_search(self, *args, **kwargs)

    table_class.search = counting_search
    try:
        yield counter
    finally:
        table_class.search = original_search


def _flush_memcached():
    client = MemcacheClient((MEMCACHED_HOST, MEMCACHED_PORT))
    client.flush_all()
    client.close()


def _make_searcher(solution_module, ttl_seconds=300):
    return solution_module.CachedSearcher(
        DB_PATH,
        TABLE_NAME,
        memcached_host=MEMCACHED_HOST,
        memcached_port=MEMCACHED_PORT,
        ttl_seconds=ttl_seconds,
    )


def _qvec(seed):
    return np.random.default_rng(seed).standard_normal(DIM).astype("float64").tolist()


def _assert_result_shape(rows):
    assert isinstance(rows, list), "search must return a list of result rows."
    prev = None
    for row in rows:
        assert set(row.keys()) == {"id", "category", "_distance"}, (
            f"Each result row must have exactly keys id, category, _distance; got {sorted(row.keys())}."
        )
        assert isinstance(row["id"], int), "Result 'id' must be an int."
        assert isinstance(row["category"], str), "Result 'category' must be a str."
        dist = float(row["_distance"])
        if prev is not None:
            assert dist >= prev - 1e-9, "Results must be ordered by ascending _distance."
        prev = dist


def test_miss_then_hit(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module)
    q = _qvec(7)

    before = search_counter["n"]
    first = searcher.search(q, 5)
    miss_delta = search_counter["n"] - before
    assert miss_delta >= 1, "First query for a fresh cache must run a LanceDB search (miss)."
    _assert_result_shape(first)

    before = search_counter["n"]
    second = searcher.search(q, 5)
    hit_delta = search_counter["n"] - before
    assert hit_delta == 0, "Repeating an identical query must be served from cache (no LanceDB search)."


def test_byte_identical_hit(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module)
    q = _qvec(7)

    miss_result = searcher.search(q, 5)
    hit_result = searcher.search(q, 5)

    _assert_result_shape(miss_result)
    assert len(hit_result) == len(miss_result), "Hit and miss results must have the same length."
    for a, b in zip(miss_result, hit_result):
        assert a["id"] == b["id"], "Cached hit ids must match the originating miss."
        assert a["category"] == b["category"], "Cached hit categories must match the originating miss."
        assert float(a["_distance"]) == float(b["_distance"]), (
            "Cached hit _distance must be identical to the originating miss."
        )


def test_different_query_misses(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module)
    q = _qvec(7)
    q2 = _qvec(99)

    searcher.search(q, 5)  # warm q

    before = search_counter["n"]
    searcher.search(q2, 5)
    assert search_counter["n"] - before >= 1, "A different query vector must be a cache miss."

    before = search_counter["n"]
    searcher.search(q2, 5)
    assert search_counter["n"] - before == 0, "Repeating the second query must be a cache hit."


def test_filter_is_part_of_key(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module)
    q = _qvec(7)

    before = search_counter["n"]
    res_a = searcher.search(q, 5, filter="category = 'A'")
    assert search_counter["n"] - before >= 1, "First filtered query must be a miss."
    _assert_result_shape(res_a)
    assert all(r["category"] == "A" for r in res_a), "Filter category='A' must return only category A rows."

    before = search_counter["n"]
    res_b = searcher.search(q, 5, filter="category = 'B'")
    assert search_counter["n"] - before >= 1, "A different filter must produce a different cache key (miss)."
    assert all(r["category"] == "B" for r in res_b), "Filter category='B' must return only category B rows."

    before = search_counter["n"]
    searcher.search(q, 5, filter="category = 'A'")
    assert search_counter["n"] - before == 0, "Repeating an identical filtered query must be a hit."


def test_invalidation_on_insert(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module)
    q3 = _qvec(123)
    new_id = 10_000_000

    warm = searcher.search(q3, 5)
    before = search_counter["n"]
    searcher.search(q3, 5)
    assert search_counter["n"] - before == 0, "Query must be cached before the insert."

    v0 = int(searcher.current_version())
    searcher.add([{"id": new_id, "category": "A", "vector": q3}])
    assert int(searcher.current_version()) > v0, "add() must bump the dataset version."

    before = search_counter["n"]
    after = searcher.search(q3, 5)
    assert search_counter["n"] - before >= 1, "After an insert the same query must recompute (miss)."
    assert after[0]["id"] == new_id, "The newly inserted identical-vector row must be the top-1 result."
    assert after != warm, "Post-insert result must differ from the stale cached result."


def test_cross_instance_invalidation(solution_module, search_counter):
    _flush_memcached()
    q = _qvec(7)
    new_id = 10_000_001

    reader = _make_searcher(solution_module)
    writer = _make_searcher(solution_module)

    reader.search(q, 5)
    before = search_counter["n"]
    reader.search(q, 5)
    assert search_counter["n"] - before == 0, "Reader must cache its query before the external write."

    writer.add([{"id": new_id, "category": "A", "vector": q}])

    before = search_counter["n"]
    result = reader.search(q, 5)
    assert search_counter["n"] - before >= 1, (
        "A write from a different instance must invalidate the reader's cache "
        "(the version counter must be shared via memcached, not kept in process memory)."
    )
    assert result[0]["id"] == new_id, "Reader must observe the row written by the other instance as top-1."


def test_invalidation_on_update(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module)
    q = _qvec(7)

    searcher.search(q, 5)
    before = search_counter["n"]
    searcher.search(q, 5)
    assert search_counter["n"] - before == 0, "Query must be cached before the update."

    v0 = int(searcher.current_version())
    searcher.update(where="id = 1", values={"category": "Z"})
    assert int(searcher.current_version()) > v0, "update() must bump the dataset version."

    before = search_counter["n"]
    searcher.search(q, 5)
    assert search_counter["n"] - before >= 1, "After an update the same query must recompute (miss)."


def test_ttl_expiry(solution_module, search_counter):
    _flush_memcached()
    searcher = _make_searcher(solution_module, ttl_seconds=1)
    q = _qvec(7)

    searcher.search(q, 5)
    before = search_counter["n"]
    searcher.search(q, 5)
    assert search_counter["n"] - before == 0, "Immediate repeat must hit the cache."

    time.sleep(2.5)

    before = search_counter["n"]
    searcher.search(q, 5)
    assert search_counter["n"] - before >= 1, (
        "After the TTL expires (with no intervening writes) the query must recompute (miss)."
    )
