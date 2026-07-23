import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

# --- Canonical configuration (must match the environment the agent developed against) ---
PROJECT_DIR = "/home/user/myproject"
NATS_URL = "nats://127.0.0.1:4222"
STREAM = "DOCS"
SUBJECT = "docs.ingest"
DURABLE = "indexer"
LANCEDB_PATH = os.path.join(PROJECT_DIR, "lancedb")
BATCH_SIZE = "10"

# ZEALT_RUN_ID is provided by the platform; never override it. Fall back for local runs.
RUN_ID = os.environ.get("ZEALT_RUN_ID", "zrlocal")
TABLE_NAME = f"documents_{RUN_ID}"

# Publish the config the solution reads. Doing this before importing the solution
# guarantees the candidate worker and this verifier agree on stream/table/paths.
os.environ["NATS_URL"] = NATS_URL
os.environ["JS_STREAM"] = STREAM
os.environ["JS_SUBJECT"] = SUBJECT
os.environ["JS_DURABLE"] = DURABLE
os.environ["LANCEDB_PATH"] = LANCEDB_PATH
os.environ["INDEX_BATCH_SIZE"] = BATCH_SIZE
os.environ["ZEALT_RUN_ID"] = RUN_ID

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

NUM_DOCS = 200
DIM = 32
RNG_SEED = 2026

_rng = np.random.default_rng(RNG_SEED)
VECTORS = _rng.standard_normal((NUM_DOCS, DIM)).astype(np.float32)


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="session")
def nats_server():
    """Ensure a local NATS server with JetStream is running on 127.0.0.1:4222."""
    started_proc = None
    if not _port_open("127.0.0.1", 4222):
        os.makedirs("/tmp/nats-store", exist_ok=True)
        started_proc = subprocess.Popen(
            ["nats-server", "-js", "-a", "127.0.0.1", "-p", "4222",
             "-sd", "/tmp/nats-store"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline and not _port_open("127.0.0.1", 4222):
            time.sleep(0.5)
        assert _port_open("127.0.0.1", 4222), "Failed to start local nats-server."
    yield
    if started_proc is not None:
        started_proc.terminate()
        try:
            started_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            started_proc.kill()


def _load_solution():
    try:
        if "solution" in sys.modules:
            return importlib.reload(sys.modules["solution"])
        return importlib.import_module("solution")
    except Exception as exc:  # pragma: no cover - clear failure message
        raise AssertionError(
            f"Could not import {PROJECT_DIR}/solution.py: {exc!r}"
        )


def _read_table_state():
    """Return (ids, vectors) currently stored in the LanceDB table, or None."""
    import lancedb

    db = lancedb.connect(LANCEDB_PATH)
    if TABLE_NAME not in db.table_names():
        return None
    tbl = db.open_table(TABLE_NAME)
    data = tbl.to_arrow().to_pydict()
    ids = [int(i) for i in data["id"]]
    vecs = [np.asarray(v, dtype=np.float32) for v in data["vector"]]
    return ids, vecs


def _assert_ids_and_vectors(state, expected_ids):
    assert state is not None, f"LanceDB table {TABLE_NAME!r} does not exist."
    ids, vecs = state
    assert len(ids) == len(expected_ids), (
        f"Expected {len(expected_ids)} rows, found {len(ids)}."
    )
    assert len(set(ids)) == len(ids), (
        f"Duplicate ids present; distinct={len(set(ids))} total={len(ids)}."
    )
    assert set(ids) == set(expected_ids), (
        f"Stored id set does not match expected. "
        f"missing={sorted(set(expected_ids) - set(ids))[:10]} "
        f"unexpected={sorted(set(ids) - set(expected_ids))[:10]}"
    )
    vec_by_id = {i: v for i, v in zip(ids, vecs)}
    for i in expected_ids:
        assert np.allclose(vec_by_id[i], VECTORS[i], atol=1e-5), (
            f"Vector for id {i} does not match the published vector."
        )


async def _connect():
    import nats

    return await nats.connect(NATS_URL, connect_timeout=10)


async def _reset_and_publish(ids):
    nc = await _connect()
    try:
        js = nc.jetstream()
        # Fresh stream state.
        try:
            await js.delete_stream(STREAM)
        except Exception:
            pass
        await js.add_stream(name=STREAM, subjects=[SUBJECT])
        for i in ids:
            body = json.dumps(
                {"id": int(i), "text": f"doc-{i}", "vector": VECTORS[i].tolist()}
            ).encode("utf-8")
            await js.publish(SUBJECT, body)
    finally:
        await nc.close()


async def _publish(ids):
    nc = await _connect()
    try:
        js = nc.jetstream()
        for i in ids:
            body = json.dumps(
                {"id": int(i), "text": f"doc-{i}", "vector": VECTORS[i].tolist()}
            ).encode("utf-8")
            await js.publish(SUBJECT, body)
    finally:
        await nc.close()


async def _consumer_info():
    nc = await _connect()
    try:
        js = nc.jetstream()
        return await js.consumer_info(STREAM, DURABLE)
    finally:
        await nc.close()


async def _drop_table():
    import lancedb

    db = lancedb.connect(LANCEDB_PATH)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)


async def _scenario(solution):
    results = {}

    # --- Setup: clean slate, then publish 200 documents in id order. ---
    await _drop_table()
    await _reset_and_publish(range(NUM_DOCS))

    # Step 1: partial run of 80 messages.
    stats1 = await solution.run_indexer(max_messages=80)
    assert isinstance(stats1, dict) and "committed" in stats1, (
        f"run_indexer must return a dict with a 'committed' key, got {stats1!r}."
    )
    assert int(stats1["committed"]) == 80, (
        f"Partial run should commit 80 messages, got {stats1!r}."
    )
    _assert_ids_and_vectors(_read_table_state(), list(range(80)))

    # Step 2: simulated restart -> drain remaining messages.
    stats2 = await solution.run_indexer()
    assert int(stats2["committed"]) == 120, (
        f"Resume run should commit the remaining 120 messages, got {stats2!r}."
    )
    _assert_ids_and_vectors(_read_table_state(), list(range(NUM_DOCS)))

    # Step 3: durable consumer state fully acknowledged.
    await asyncio.sleep(1.0)  # allow async acks to reach the server
    ci = await _consumer_info()
    assert ci.num_pending == 0, (
        f"Expected 0 pending messages after full drain, got {ci.num_pending}."
    )
    assert ci.num_ack_pending == 0, (
        f"Expected 0 ack-pending messages after full drain, got {ci.num_ack_pending}."
    )
    assert ci.ack_floor.stream_seq == NUM_DOCS, (
        f"Expected ack floor stream_seq == {NUM_DOCS}, got {ci.ack_floor.stream_seq}."
    )

    # Step 4: drained no-op.
    stats3 = await solution.run_indexer()
    assert int(stats3["committed"]) == 0, (
        f"Running against a drained stream should commit 0, got {stats3!r}."
    )
    _assert_ids_and_vectors(_read_table_state(), list(range(NUM_DOCS)))

    # Step 5: redelivery / exactly-once effect. Re-publish ids 0..49.
    await _publish(range(50))
    stats4 = await solution.run_indexer()
    assert int(stats4["committed"]) == 50, (
        f"Re-published 50 messages should be consumed, got {stats4!r}."
    )
    # Despite consuming duplicates, the table must still hold exactly 200 unique ids
    # with unchanged vectors (idempotent upsert-by-id).
    _assert_ids_and_vectors(_read_table_state(), list(range(NUM_DOCS)))

    await asyncio.sleep(1.0)
    ci2 = await _consumer_info()
    assert ci2.num_pending == 0 and ci2.num_ack_pending == 0, (
        f"After consuming the re-published batch, consumer must be fully acked; "
        f"got pending={ci2.num_pending} ack_pending={ci2.num_ack_pending}."
    )

    results["ok"] = True
    return results


def test_durable_exactly_once_indexing(nats_server):
    solution = _load_solution()
    assert hasattr(solution, "run_indexer"), (
        "solution.py must expose an async 'run_indexer' coroutine function."
    )
    asyncio.run(_scenario(solution))
