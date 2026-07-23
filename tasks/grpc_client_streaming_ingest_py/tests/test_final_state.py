import os
import sys
import socket
import importlib

import numpy as np
import pytest
from xprocess import ProcessStarter

PROJECT_DIR = "/home/user/myproject"
HOST = "127.0.0.1"
PORT = 50051
ADDRESS = f"{HOST}:{PORT}"

LANCEDB_PATH = os.environ.get("LANCEDB_PATH", os.path.join(PROJECT_DIR, "lance_data"))
TABLE_NAME = f"vectors_{os.environ.get('ZEALT_RUN_ID', '')}"

DIM = 16
NUM_RECORDS = 250
BATCH_SIZE = 100
EXPECTED_BATCHES = 3  # ceil(250 / 100)

# Ensure the candidate's project (server.py, client.py, generated stubs) is importable.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _make_records():
    """Deterministic record set streamed to the server."""
    rng = np.random.default_rng(20260715)
    records = []
    for i in range(NUM_RECORDS):
        vec = rng.standard_normal(DIM).astype("float32")
        records.append({"id": i, "vector": vec.tolist(), "metadata": f"rec-{i}"})
    return records


def _load_client():
    """Import the candidate's client helper module fresh."""
    if "client" in sys.modules:
        importlib.reload(sys.modules["client"])
        return sys.modules["client"]
    return importlib.import_module("client")


def _open_table():
    import lancedb

    db = lancedb.connect(LANCEDB_PATH)
    assert TABLE_NAME in db.table_names(), (
        f"LanceDB table '{TABLE_NAME}' not found at {LANCEDB_PATH}. "
        f"Available tables: {db.table_names()}"
    )
    return db.open_table(TABLE_NAME)


def _read_all_rows():
    tbl = _open_table()
    df = tbl.to_pandas()
    rows = {}
    for _, r in df.iterrows():
        rows[int(r["id"])] = {
            "vector": np.asarray(r["vector"], dtype="float32"),
            "metadata": str(r["metadata"]),
        }
    return rows


@pytest.fixture(scope="session")
def grpc_server(xprocess):
    class Starter(ProcessStarter):
        name = "grpc_server"
        args = [sys.executable, "server.py"]
        env = os.environ.copy()
        popen_kwargs = {
            "cwd": PROJECT_DIR,
            "text": True,
        }
        timeout = 120
        terminate_on_interrupt = True

        def startup_check(self):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                return s.connect_ex((HOST, PORT)) == 0

    info = xprocess.getinfo(Starter.name)
    printed = 0

    def capture_logs(tag):
        nonlocal printed
        try:
            with open(info.logpath, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []
        new = lines[printed:]
        printed = len(lines)
        print(f"===== [{tag}] {Starter.name} log begin =====")
        print("".join(new))
        print(f"===== [{tag}] {Starter.name} log end =====")

    started = False
    try:
        xprocess.ensure(Starter.name, Starter)
        started = True
    finally:
        capture_logs("STARTED" if started else "FAILED")

    yield ADDRESS

    capture_logs("TEARDOWN")
    # Terminate so each run starts from a fresh, empty table.
    info.terminate()


def test_client_streaming_ingest_summary(grpc_server):
    client = _load_client()
    records = _make_records()
    summary = client.ingest_vectors(records, address=grpc_server)
    assert isinstance(summary, dict), f"ingest_vectors must return a dict, got {type(summary)}"
    assert int(summary["received"]) == NUM_RECORDS, (
        f"Expected received={NUM_RECORDS}, got {summary.get('received')}"
    )
    assert int(summary["written"]) == NUM_RECORDS, (
        f"Expected written={NUM_RECORDS}, got {summary.get('written')}"
    )
    assert int(summary["batches"]) == EXPECTED_BATCHES, (
        f"Expected batches={EXPECTED_BATCHES} (batch size {BATCH_SIZE}), got {summary.get('batches')}"
    )


def test_exactly_once_persistence(grpc_server):
    records = _make_records()
    rows = _read_all_rows()
    assert len(rows) == NUM_RECORDS, (
        f"Expected exactly {NUM_RECORDS} rows in the table, found {len(rows)} "
        f"(possible duplicate or missing writes)."
    )
    assert set(rows.keys()) == set(range(NUM_RECORDS)), (
        "Stored id set does not match the ingested id set {0..249}."
    )
    for rec in records:
        rid = rec["id"]
        stored = rows[rid]
        assert stored["metadata"] == rec["metadata"], (
            f"metadata mismatch for id {rid}: expected {rec['metadata']!r}, got {stored['metadata']!r}"
        )
        assert np.allclose(stored["vector"], np.asarray(rec["vector"], dtype="float32"), atol=1e-5), (
            f"Stored vector for id {rid} does not match the ingested vector."
        )


def test_unary_search_matches_bruteforce(grpc_server):
    client = _load_client()
    records = _make_records()

    ids = np.array([r["id"] for r in records])
    mat = np.stack([np.asarray(r["vector"], dtype="float32") for r in records])

    rng2 = np.random.default_rng(777)
    q = rng2.standard_normal(DIM).astype("float32")

    dists = np.sum((mat - q) ** 2, axis=1)
    order = sorted(range(len(records)), key=lambda i: (float(dists[i]), int(ids[i])))
    expected_top5 = [int(ids[i]) for i in order[:5]]

    hits = client.search(q.tolist(), 5, address=grpc_server)
    assert isinstance(hits, list), f"search must return a list, got {type(hits)}"
    assert len(hits) == 5, f"Expected 5 hits, got {len(hits)}"

    got_ids = [int(h["id"]) for h in hits]
    assert got_ids == expected_top5, (
        f"Search top-5 ids {got_ids} do not match brute-force ground truth {expected_top5}."
    )

    got_dists = [float(h["distance"]) for h in hits]
    assert all(got_dists[i] <= got_dists[i + 1] + 1e-6 for i in range(len(got_dists) - 1)), (
        f"Search distances are not non-decreasing: {got_dists}"
    )

    for h in hits:
        assert set(["id", "distance", "metadata"]).issubset(h.keys()), (
            f"Each hit must contain keys id, distance, metadata; got {list(h.keys())}"
        )
        assert h["metadata"] == f"rec-{int(h['id'])}", (
            f"Hit metadata mismatch for id {h['id']}: {h['metadata']!r}"
        )


def test_wrong_dimension_rejected_without_corruption(grpc_server):
    import grpc

    client = _load_client()

    count_before = len(_read_all_rows())
    assert count_before == NUM_RECORDS, (
        f"Precondition failed: expected {NUM_RECORDS} rows before bad ingest, got {count_before}."
    )

    rng = np.random.default_rng(4242)
    bad_records = []
    for j in range(10):
        rid = 1000 + j
        dim = 15 if j == 3 else DIM  # record index 3 has the wrong dimension
        vec = rng.standard_normal(dim).astype("float32")
        bad_records.append({"id": rid, "vector": vec.tolist(), "metadata": f"bad-{rid}"})

    with pytest.raises(grpc.RpcError) as excinfo:
        client.ingest_vectors(bad_records, address=grpc_server)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT, (
        f"Expected gRPC status INVALID_ARGUMENT for a wrong-dimension record, "
        f"got {excinfo.value.code()}"
    )

    rows_after = _read_all_rows()
    assert len(rows_after) == NUM_RECORDS, (
        f"Table row count changed after a rejected ingest: expected {NUM_RECORDS}, got {len(rows_after)} "
        f"(the aborted RPC must not leave a partial write)."
    )
    leaked = [rid for rid in range(1000, 1010) if rid in rows_after]
    assert not leaked, (
        f"Ids from the rejected stream leaked into the table: {leaked}"
    )
