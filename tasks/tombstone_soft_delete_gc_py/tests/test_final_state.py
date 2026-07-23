import os
import sys
import time

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
DB_PATH = os.path.join(PROJECT_DIR, "lancedb_data")
DIM = 16
N = 60

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "").strip()
    return f"tombstones_{run_id}" if run_id else "tombstones_local"


def _make_docs():
    rng = np.random.default_rng(2026)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    docs = []
    for i in range(N):
        docs.append(
            {"id": i, "text": f"document number {i}", "vector": vecs[i].tolist()}
        )
    return docs, vecs


def _open_raw(name):
    """Open a *fresh* connection so the latest committed version is always read."""
    import lancedb

    conn = lancedb.connect(DB_PATH)
    return conn.open_table(name)


def _brute_force_topk(query, vecs, live_ids, k):
    q = np.asarray(query, dtype=np.float32)
    ordered = sorted(
        live_ids,
        key=lambda i: (float(np.sum((vecs[i] - q) ** 2)), int(i)),
    )
    return ordered[:k]


@pytest.fixture(scope="module")
def env():
    """Seed the table once and expose helpers for the lifecycle test."""
    try:
        import solution  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Unable to import solution module from {PROJECT_DIR}: {exc}")

    from solution import TombstoneStore

    # Start from a clean state.
    import shutil

    if os.path.isdir(DB_PATH):
        shutil.rmtree(DB_PATH)

    name = _table_name()
    docs, vecs = _make_docs()

    store = TombstoneStore(name)
    # Insert in four batches of 15 to create several physical fragments.
    for start in range(0, N, 15):
        store.add_documents(docs[start : start + 15])

    return {
        "TombstoneStore": TombstoneStore,
        "store": store,
        "name": name,
        "docs": docs,
        "vecs": vecs,
    }


def test_full_lifecycle(env):
    TombstoneStore = env["TombstoneStore"]
    store = env["store"]
    name = env["name"]
    vecs = env["vecs"]

    # ---- Step 1: baseline search shape ------------------------------------
    q1 = vecs[30].tolist()
    res1 = store.search(q1, 5)
    assert isinstance(res1, list) and len(res1) == 5, (
        f"Expected 5 baseline search results, got {res1!r}"
    )
    for r in res1:
        assert set(r.keys()) == {"id", "text", "distance"}, (
            f"Each search result must have exactly keys id, text, distance; got {set(r.keys())}"
        )
    assert int(res1[0]["id"]) == 30, (
        f"Row queried with its own vector should rank first; got id {res1[0]['id']}"
    )
    dists1 = [float(r["distance"]) for r in res1]
    assert all(dists1[i] <= dists1[i + 1] + 1e-6 for i in range(len(dists1) - 1)), (
        f"Distances must be non-decreasing; got {dists1}"
    )

    # ---- Step 2: soft delete hides but physically retains -----------------
    changed = store.soft_delete([0, 1, 2, 3, 4])
    assert int(changed) == 5, f"soft_delete of 5 ids should return 5; got {changed}"

    res2 = store.search(vecs[0].tolist(), 10)
    got_ids = {int(r["id"]) for r in res2}
    assert got_ids.isdisjoint({0, 1, 2, 3, 4}), (
        f"Tombstoned ids must not appear in search results; got {sorted(got_ids)}"
    )

    raw = _open_raw(name)
    assert raw.count_rows() == 60, (
        f"Soft delete must not physically remove rows; expected 60, got {raw.count_rows()}"
    )
    assert raw.count_rows("deleted = true") == 5, (
        f"Expected 5 tombstoned rows, got {raw.count_rows('deleted = true')}"
    )

    # ---- Step 3: restore un-tombstones ------------------------------------
    restored = store.restore([0, 1])
    assert int(restored) == 2, f"restore of 2 ids should return 2; got {restored}"

    # Each restored row queried with its own vector must reappear at rank 1.
    for rid in (0, 1):
        res3 = store.search(vecs[rid].tolist(), 5)
        assert int(res3[0]["id"]) == rid, (
            f"Restored id {rid} should be searchable again and rank first for its own "
            f"vector; got {[int(r['id']) for r in res3]}"
        )

    raw = _open_raw(name)
    assert raw.count_rows("deleted = true") == 3, (
        f"After restoring 2, expected 3 tombstoned rows, got {raw.count_rows('deleted = true')}"
    )
    df = raw.to_pandas()
    da = {int(row.id): int(row.deleted_at) for row in df.itertuples()}
    assert da[0] == 0 and da[1] == 0, (
        f"Restored rows must have deleted_at reset to 0; got 0->{da[0]}, 1->{da[1]}"
    )

    # ---- Step 4: set up aged vs fresh tombstones --------------------------
    re_deleted = store.soft_delete([0, 1])
    assert int(re_deleted) == 2, (
        f"Re-tombstoning ids 0 and 1 should return 2; got {re_deleted}"
    )

    now = int(time.time())
    raw = _open_raw(name)
    raw.update(where="id IN (0, 1, 2)", values={"deleted_at": now - 1_000_000})

    # ---- Step 5: capture pre-gc physical state ----------------------------
    raw = _open_raw(name)
    pre_fragments = len(raw.to_lance().get_fragments())
    pre_versions = len(raw.list_versions())
    assert pre_fragments >= 2, (
        f"Expected several fragments before gc, got {pre_fragments}"
    )

    # ---- Step 6: garbage collect ------------------------------------------
    gc_store = TombstoneStore(name)  # fresh handle sees the aged deleted_at values
    hard_deleted = gc_store.gc(86400)
    assert int(hard_deleted) == 3, (
        f"gc should hard-delete the 3 aged tombstones (ids 0,1,2); got {hard_deleted}"
    )

    raw = _open_raw(name)
    assert raw.count_rows() == 57, (
        f"After gc, 3 aged rows must be physically gone (expected 57 rows), got {raw.count_rows()}"
    )
    assert raw.count_rows("id IN (0, 1, 2)") == 0, (
        "Aged tombstones (ids 0,1,2) must be physically removed after gc."
    )
    assert raw.count_rows("deleted = true") == 2, (
        f"Non-aged tombstones (ids 3,4) must remain; expected 2, got {raw.count_rows('deleted = true')}"
    )

    df = raw.to_pandas()
    by_id = {int(row.id): np.asarray(row.vector, dtype=np.float32) for row in df.itertuples()}
    for i in range(5, 60):
        assert i in by_id, f"Live row id {i} must still be present after gc."
        assert np.allclose(by_id[i], vecs[i], atol=1e-5), (
            f"Vector for live row id {i} must be unchanged after gc."
        )

    post_fragments = len(raw.to_lance().get_fragments())
    post_versions = len(raw.list_versions())
    assert 1 <= post_fragments < pre_fragments, (
        f"Fragment count must strictly shrink after compaction: pre={pre_fragments}, post={post_fragments}"
    )
    assert 1 <= post_versions < pre_versions, (
        f"Version history must strictly shrink after cleanup: pre={pre_versions}, post={post_versions}"
    )

    # ---- Step 7: search correctness after gc ------------------------------
    live_ids = list(range(5, 60))
    qrng = np.random.default_rng(99)
    for _ in range(5):
        query = qrng.standard_normal(DIM).astype(np.float32)
        expected = _brute_force_topk(query, vecs, live_ids, 5)
        res = gc_store.search(query.tolist(), 5)
        got = [int(r["id"]) for r in res]
        assert got == expected, (
            f"search must return brute-force L2 top-5 over live rows; expected {expected}, got {got}"
        )
        assert set(got).isdisjoint({0, 1, 2, 3, 4}), (
            f"search must never return tombstoned or deleted ids; got {got}"
        )
