import importlib
import json
import os
import shutil
import sys
import threading
import time
from datetime import timedelta

import numpy as np
import pytest


PROJECT_DIR = "/home/user/myproject"
DB_PATH = "/home/user/myproject/lancedb_data"
POINTER_PATH = "/app/index_pointer.json"


@pytest.fixture(scope="session")
def env():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID must be set"
    base = f"vectors_{run_id}"
    return {"run_id": run_id, "base": base}


@pytest.fixture(scope="session")
def reset_pointer_and_shadow(env):
    """Reset pointer to base and remove any leftover shadow table from prior runs."""
    import lancedb

    shadow_name = f"{env['base']}_shadow_{env['run_id']}"
    shadow_dir = os.path.join(DB_PATH, f"{shadow_name}.lance")
    if os.path.isdir(shadow_dir):
        shutil.rmtree(shadow_dir, ignore_errors=True)

    with open(POINTER_PATH, "w") as f:
        json.dump({"active": env["base"]}, f)

    # Sanity: base table must exist with an index.
    db = lancedb.connect(DB_PATH)
    assert env["base"] in db.table_names(), f"Base table {env['base']} missing"
    yield shadow_name


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Force a fresh import so any leftover module state is dropped.
    if "solution" in sys.modules:
        del sys.modules["solution"]
    mod = importlib.import_module("solution")
    return mod


def test_solution_exports_class(solution_module):
    assert hasattr(solution_module, "IndexSwapManager"), \
        "solution.py must define class IndexSwapManager"
    cls = solution_module.IndexSwapManager
    for name in ("build_shadow_index", "search", "promote_shadow"):
        assert callable(getattr(cls, name, None)), \
            f"IndexSwapManager must define a callable method {name!r}"


def _percentile(values, p):
    if not values:
        return float("inf")
    arr = sorted(values)
    k = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
    return arr[k]


def test_zero_downtime_swap(env, reset_pointer_and_shadow, solution_module):
    shadow_name_expected = reset_pointer_and_shadow

    mgr = solution_module.IndexSwapManager(env["base"])

    latencies = []
    errors = []
    result_lengths = []
    stop_flag = threading.Event()

    def worker():
        rng_seeds = list(range(7000, 7100))
        for i, seed in enumerate(rng_seeds):
            if stop_flag.is_set():
                break
            qv = np.random.default_rng(seed).standard_normal(64).astype(np.float32)
            t0 = time.perf_counter()
            try:
                result = mgr.search(qv, 5)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                result_lengths.append(len(result) if result is not None else 0)
            except Exception as exc:  # noqa: BLE001
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                errors.append(repr(exc))
                result_lengths.append(0)
            # Pace ~100 ms between calls -> 100 calls in ~10s.
            elapsed = time.perf_counter() - t0
            remaining = 0.1 - elapsed
            if remaining > 0:
                time.sleep(remaining)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Let a few searches hit the base table before mutating.
    time.sleep(0.5)

    shadow_name = mgr.build_shadow_index(
        "IVF_PQ", {"num_partitions": 4, "num_sub_vectors": 4}
    )
    assert isinstance(shadow_name, str) and shadow_name, \
        f"build_shadow_index must return the shadow table name string, got {shadow_name!r}"
    assert shadow_name.endswith(f"_shadow_{env['run_id']}"), (
        f"Shadow table name must end with _shadow_{env['run_id']}, got {shadow_name!r}"
    )
    assert shadow_name == shadow_name_expected, (
        f"Expected shadow name {shadow_name_expected!r}, manager returned {shadow_name!r}"
    )

    mgr.promote_shadow()

    t.join(timeout=20)
    assert not t.is_alive(), "Background search worker did not finish in time."

    # Acceptance: every search succeeded with non-empty results.
    assert errors == [], f"Background searches raised exceptions: {errors[:3]}"
    assert len(latencies) == 100, f"Expected 100 search calls, got {len(latencies)}"
    assert all(n > 0 for n in result_lengths), (
        f"Every search must return a non-empty list; got result_lengths counts="
        f"{{nonzero={sum(1 for n in result_lengths if n>0)}, zero={sum(1 for n in result_lengths if n==0)}}}"
    )

    p99 = _percentile(latencies, 99)
    assert p99 < 500.0, f"p99 latency must be < 500 ms, got {p99:.1f} ms"


def test_pointer_promoted(env, reset_pointer_and_shadow):
    expected_shadow = reset_pointer_and_shadow
    with open(POINTER_PATH) as f:
        data = json.load(f)
    assert data.get("active") == expected_shadow, (
        f"Pointer file should reference {expected_shadow!r} after promote, got {data!r}"
    )


def test_shadow_table_has_new_index(env, reset_pointer_and_shadow):
    import lancedb

    shadow = reset_pointer_and_shadow
    db = lancedb.connect(DB_PATH)
    assert shadow in db.table_names(), (
        f"Shadow table {shadow!r} was not created on disk. Existing: {db.table_names()}"
    )
    tbl = db.open_table(shadow)
    assert tbl.count_rows() >= 300, (
        f"Shadow table must contain >=300 rows, got {tbl.count_rows()}"
    )
    indices = tbl.list_indices()
    vector_indices = [i for i in indices if list(i.columns) == ["vector"]]
    assert len(vector_indices) >= 1, (
        f"Shadow table must expose a vector index, got list_indices()={indices!r}"
    )
    # Wait briefly in case the index is still being committed (timedelta required).
    try:
        tbl.wait_for_index([vector_indices[0].name], timeout=timedelta(seconds=30))
    except Exception:
        pass
    stats = tbl.index_stats(vector_indices[0].name)
    num_indexed = getattr(stats, "num_indexed_rows", None)
    num_unindexed = getattr(stats, "num_unindexed_rows", None)
    if num_indexed is None and isinstance(stats, dict):
        num_indexed = stats.get("num_indexed_rows")
        num_unindexed = stats.get("num_unindexed_rows")
    assert num_indexed is not None and num_indexed >= 300, (
        f"Shadow index must report num_indexed_rows>=300, stats={stats!r}"
    )
    assert num_unindexed == 0, (
        f"Shadow index must have num_unindexed_rows==0, stats={stats!r}"
    )


def test_post_promote_search_hits_shadow(env, reset_pointer_and_shadow, solution_module):
    import lancedb

    shadow = reset_pointer_and_shadow
    db = lancedb.connect(DB_PATH)
    shadow_tbl = db.open_table(shadow)
    shadow_ids = set(int(x) for x in shadow_tbl.to_pandas()["id"].tolist())

    mgr = solution_module.IndexSwapManager(env["base"])
    results = mgr.search(np.zeros(64, dtype=np.float32), 5)
    assert results, "Post-promote search must return at least one row."
    for r in results:
        rid = r["id"] if isinstance(r, dict) else r.get("id")
        assert int(rid) in shadow_ids, (
            f"Returned id {rid} is not present in the shadow table id set."
        )
