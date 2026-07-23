import base64
import importlib.util
import os
import subprocess
import sys
import time
from datetime import timedelta

import pytest
import requests

PROJECT_DIR = "/home/user/myproject"
DB_URI = "/home/user/myproject/lancedb"
TABLE_NAME = "documents"
ETCD_URL = "http://127.0.0.1:2379"
ELECTION_PREFIX = "/lancedb/indexer/"
TTL = 3

# Shared, ordered scenario state passed between the ordered test functions.
CTX = {}


# --------------------------------------------------------------------------- #
# etcd JSON gRPC-gateway helpers
# --------------------------------------------------------------------------- #
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _etcd_post(path: str, body: dict) -> requests.Response:
    return requests.post(f"{ETCD_URL}{path}", json=body, timeout=10)


def _ensure_etcd_running() -> bool:
    for _ in range(3):
        try:
            resp = _etcd_post("/v3/kv/range", {"key": _b64(b"\x00")})
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        starter = "/usr/local/bin/start-etcd.sh"
        if os.path.isfile(starter):
            subprocess.run(["bash", starter], check=False)
            time.sleep(3)
    return False


def _etcd_clear_prefix(prefix: str) -> None:
    key = prefix.encode()
    # range_end for a prefix scan = key with its last byte incremented.
    range_end = key[:-1] + bytes([key[-1] + 1])
    _etcd_post(
        "/v3/kv/deleterange",
        {"key": _b64(key), "range_end": _b64(range_end)},
    )


# --------------------------------------------------------------------------- #
# LanceDB helpers
# --------------------------------------------------------------------------- #
def _open_table():
    import lancedb

    db = lancedb.connect(DB_URI)
    return db.open_table(TABLE_NAME)


def _index_name(tbl) -> str:
    indices = tbl.list_indices()
    assert len(indices) >= 1, "Vector index missing from the shared table."
    return indices[0].name


def _num_unindexed(tbl) -> int:
    stats = tbl.index_stats(_index_name(tbl))
    assert stats is not None, "index_stats returned None for the shared table."
    return int(stats.num_unindexed_rows)


def _num_unindexed_expect_zero(tbl, attempts: int = 10) -> int:
    val = _num_unindexed(tbl)
    for _ in range(attempts):
        if val == 0:
            return 0
        try:
            tbl.wait_for_index([_index_name(tbl)], timedelta(seconds=15))
        except Exception:
            time.sleep(1)
        val = _num_unindexed(tbl)
    return val


def _num_versions(tbl) -> int:
    return len(tbl.list_versions())


# --------------------------------------------------------------------------- #
# Candidate solution loader
# --------------------------------------------------------------------------- #
def _load_coordinator_cls():
    sol_path = os.path.join(PROJECT_DIR, "solution.py")
    assert os.path.isfile(sol_path), f"Candidate solution not found at {sol_path}."
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("candidate_solution", sol_path)
    assert spec is not None and spec.loader is not None, "Cannot load solution.py."
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "IndexerCoordinator"), (
        "solution.py must define a class named 'IndexerCoordinator'."
    )
    return module.IndexerCoordinator


# --------------------------------------------------------------------------- #
# Module-scoped environment setup: etcd + fresh dataset
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", autouse=True)
def environment():
    assert _ensure_etcd_running(), f"Local etcd not reachable at {ETCD_URL}."
    _etcd_clear_prefix(ELECTION_PREFIX)

    # Reset the shared dataset to the known seeded state (300 indexed + 200
    # unindexed rows). Run in a separate process to avoid lancedb teardown noise.
    result = subprocess.run(
        [sys.executable, "seed_dataset.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"seed_dataset.py failed (rc={result.returncode}).\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    tbl = _open_table()
    CTX["Coordinator"] = _load_coordinator_cls()
    CTX["baseline_rows"] = tbl.count_rows()
    CTX["baseline_versions"] = _num_versions(tbl)
    CTX["coordinators"] = []

    yield

    for coord in CTX.get("coordinators", []):
        try:
            coord.close()
        except Exception:
            pass
    _etcd_clear_prefix(ELECTION_PREFIX)


def _new_coordinator(worker_id: str):
    coord = CTX["Coordinator"](worker_id, ttl=TTL)
    CTX["coordinators"].append(coord)
    return coord


# --------------------------------------------------------------------------- #
# Ordered scenario
# --------------------------------------------------------------------------- #
def test_01_single_leader_wins():
    a = _new_coordinator("A")
    b = _new_coordinator("B")

    assert a.acquire() is True, "First coordinator A should win leadership."
    assert a.is_leader() is True, "A should report itself as the leader after acquire()."
    token_a = a.fencing_token()
    assert isinstance(token_a, int) and token_a > 0, (
        f"A.fencing_token() must be a positive int, got {token_a!r}."
    )

    assert b.acquire() is False, "B must not win leadership while A holds it."
    assert b.is_leader() is False, "B should not report itself as leader."
    assert b.fencing_token() is None, (
        "A non-leader coordinator's fencing_token() must be None."
    )

    CTX["a"] = a
    CTX["b"] = b
    CTX["token_a"] = token_a


def test_02_followers_are_fenced_out():
    b = CTX["b"]
    with pytest.raises(PermissionError):
        b.run_maintenance()

    tbl = _open_table()
    assert tbl.count_rows() == 500, (
        "A rejected follower must not modify the table (row count changed)."
    )
    assert _num_unindexed(tbl) == 200, (
        "A rejected follower must not run maintenance (unindexed rows changed)."
    )


def test_03_leader_runs_maintenance_with_effect():
    a = CTX["a"]
    r = a.run_maintenance()

    assert isinstance(r, dict), "run_maintenance() must return a dict."
    expected_keys = {
        "worker_id",
        "fencing_token",
        "version_before",
        "version_after",
        "unindexed_before",
        "unindexed_after",
    }
    assert set(r.keys()) == expected_keys, (
        f"run_maintenance() must return exactly keys {expected_keys}, got {set(r.keys())}."
    )
    assert r["worker_id"] == "A", f"Expected worker_id 'A', got {r['worker_id']!r}."
    assert r["fencing_token"] == CTX["token_a"], (
        "run_maintenance()'s fencing_token must match the leader's current term token."
    )
    assert r["unindexed_before"] == 200, (
        f"Expected 200 unindexed rows before maintenance, got {r['unindexed_before']}."
    )
    assert r["unindexed_after"] == 0, (
        f"Expected 0 unindexed rows after maintenance, got {r['unindexed_after']}."
    )
    assert r["version_after"] > r["version_before"], (
        "Maintenance must produce a new table version "
        f"(before={r['version_before']}, after={r['version_after']})."
    )

    # Independently confirm the real effect on the LanceDB table.
    tbl = _open_table()
    assert tbl.count_rows() == 500, (
        "Maintenance must be idempotent and lose no rows (expected 500)."
    )
    assert _num_unindexed_expect_zero(tbl) == 0, (
        "After maintenance the vector index must have 0 unindexed rows."
    )
    assert _num_versions(tbl) > CTX["baseline_versions"], (
        "Maintenance must create at least one new table version."
    )


def test_04_only_one_leader_at_a_time():
    a = CTX["a"]
    b = CTX["b"]
    assert a.is_leader() is True, "A should still be the leader."
    with pytest.raises(PermissionError):
        b.run_maintenance()
    assert b.acquire() is False, "B still must not acquire leadership while A holds it."


def test_05_graceful_failover_increasing_token():
    a = CTX["a"]
    b = CTX["b"]

    a.resign()
    assert a.is_leader() is False, "A must relinquish leadership after resign()."

    assert b.acquire() is True, "B should immediately win leadership after A resigns."
    assert b.is_leader() is True, "B should now report itself as leader."
    token_b = b.fencing_token()
    assert isinstance(token_b, int), "B.fencing_token() must be an int after acquiring."
    assert token_b > CTX["token_a"], (
        f"New term token ({token_b}) must be strictly greater than the previous "
        f"term token ({CTX['token_a']})."
    )
    CTX["token_b"] = token_b


def test_06_stale_leader_is_fenced():
    a = CTX["a"]
    # A was deposed but may still believe it is the leader in-memory.
    with pytest.raises(PermissionError):
        a.run_maintenance()


def test_07_new_leader_can_run_maintenance():
    import seed_dataset  # provided in the environment

    b = CTX["b"]
    # Introduce fresh, not-yet-indexed rows without dropping the table.
    added = seed_dataset.append_unindexed(150)
    assert added == 150

    if not b.is_leader():
        assert b.acquire() is True, "Current leader B should be able to (re)acquire."

    r = b.run_maintenance()
    assert r["worker_id"] == "B", f"Expected worker_id 'B', got {r['worker_id']!r}."
    assert r["unindexed_after"] == 0, (
        "New leader's maintenance must index all pending rows (unindexed_after == 0)."
    )
    assert r["version_after"] > r["version_before"], (
        "New leader's maintenance must create a new table version."
    )

    tbl = _open_table()
    assert tbl.count_rows() == 650, (
        f"After adding 150 rows the table should have 650 rows, got {tbl.count_rows()}."
    )
    assert _num_unindexed_expect_zero(tbl) == 0, (
        "All rows must be indexed after the new leader's maintenance."
    )


def test_08_ungraceful_ttl_failover():
    b = CTX["b"]
    b.simulate_crash()
    assert b.is_leader() is False, (
        "A crashed leader must stop reporting itself as leader."
    )

    c = _new_coordinator("C")
    # Before the lease TTL elapses the key may still be held.
    c.acquire()

    time.sleep(TTL + 3)
    assert c.acquire() is True, (
        "After the crashed leader's lease TTL expires, C must be able to take over."
    )
    assert c.is_leader() is True, "C should now report itself as leader."
    token_c = c.fencing_token()
    assert isinstance(token_c, int), "C.fencing_token() must be an int after acquiring."
    assert token_c > CTX["token_b"], (
        f"C's term token ({token_c}) must exceed the previous term token "
        f"({CTX['token_b']})."
    )


def test_09_cleanup_closes_cleanly():
    for coord in CTX.get("coordinators", []):
        coord.close()
