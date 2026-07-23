import os
import re
import subprocess
import time

import lancedb
import numpy as np
import pyarrow as pa
import pytest
import redis

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/home/user/myproject/vec_db"
CONSUMER_NAME = "worker-1"
N_MESSAGES = 250
SEED = 2026
VECTOR_DIM = 32
BATCH_SIZE = 50
N_PENDING = 7          # entries delivered to a crashed consumer, left un-acked
N_CORRUPTED = 4        # rows pre-written to LanceDB before the "crash"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

DONE_RE = re.compile(r"DONE\s+ingested=(\d+)\s+reclaimed=(\d+)")


def _run_id() -> str:
    path = "/logs/artifacts/run-id"
    if os.path.isfile(path):
        with open(path) as f:
            rid = f.read().strip()
            if rid:
                return rid
    rid = os.environ.get("ZEALT_RUN_ID", "").strip()
    if rid:
        return rid
    return "local"


RUN_ID = _run_id()
STREAM_KEY = f"estream_{RUN_ID}"
GROUP_NAME = f"grp_{RUN_ID}"
TABLE_NAME = f"embeddings_{RUN_ID}"


def _wait_for_redis(client, attempts=20):
    last = None
    for _ in range(attempts):
        try:
            if client.ping():
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1)
    raise RuntimeError(f"Local redis not reachable at {REDIS_HOST}:{REDIS_PORT}: {last}")


def _lance_schema():
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def _run_consumer(consumer_name):
    env = os.environ.copy()
    env.update(
        {
            "REDIS_HOST": REDIS_HOST,
            "REDIS_PORT": str(REDIS_PORT),
            "STREAM_KEY": STREAM_KEY,
            "GROUP_NAME": GROUP_NAME,
            "CONSUMER_NAME": consumer_name,
            "LANCEDB_DIR": LANCEDB_DIR,
            "TABLE_NAME": TABLE_NAME,
            "BATCH_SIZE": str(BATCH_SIZE),
            "VECTOR_DIM": str(VECTOR_DIM),
        }
    )
    result = subprocess.run(
        ["python3", "run_consumer.py"],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("---- run_consumer stdout ----")
    print(result.stdout)
    print("---- run_consumer stderr ----")
    print(result.stderr)
    return result


def _read_table_rows():
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(TABLE_NAME)
    return tbl.to_arrow().to_pylist()


@pytest.fixture(scope="module")
def scenario():
    """Build the crash scenario, run the candidate consumer once, and expose context."""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=5)
    _wait_for_redis(client)

    # 1. Clean any prior state so the scenario is reproducible.
    client.delete(STREAM_KEY)
    db = lancedb.connect(LANCEDB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    # 2. Seed N deterministic messages onto the stream.
    rng = np.random.default_rng(SEED)
    expected_vectors = {}
    expected_texts = {}
    last_entry_id = None
    for i in range(N_MESSAGES):
        mid = f"msg-{i:04d}"
        vec = rng.standard_normal(VECTOR_DIM).astype("<f4")
        text = f"document {i} about vectors and streams"
        expected_vectors[mid] = vec
        expected_texts[mid] = text
        last_entry_id = client.xadd(
            STREAM_KEY,
            {"id": mid, "vector": vec.tobytes(), "text": text},
        )

    # 3. Create the consumer group from the beginning of the stream.
    client.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=False)

    # 4. Simulate a crash: a throwaway consumer reads the first N_PENDING entries
    #    (they become pending / un-acked) but never acknowledges them.
    delivered = client.xreadgroup(
        GROUP_NAME, "dead-consumer", {STREAM_KEY: ">"}, count=N_PENDING
    )
    assert delivered, "Failed to deliver messages to the throwaway 'dead-consumer'."

    # 5. Emulate a partial commit before the crash: pre-write the first N_CORRUPTED
    #    rows into LanceDB with all-zero vectors and wrong text. A correct idempotent
    #    upsert must overwrite these, not duplicate them.
    corrupted_rows = [
        {"id": f"msg-{i:04d}", "text": "CORRUPTED", "vector": [0.0] * VECTOR_DIM}
        for i in range(N_CORRUPTED)
    ]
    db.create_table(TABLE_NAME, data=corrupted_rows, schema=_lance_schema())

    # 6. Run the candidate consumer (first run: must recover + drain).
    first = _run_consumer(CONSUMER_NAME)

    ctx = {
        "client": client,
        "expected_vectors": expected_vectors,
        "expected_texts": expected_texts,
        "last_entry_id": last_entry_id,
        "first": first,
    }
    return ctx


def test_first_run_reports_full_recovery(scenario):
    first = scenario["first"]
    assert first.returncode == 0, f"First consumer run failed: {first.stderr}"
    m = DONE_RE.search(first.stdout)
    assert m, f"Expected a 'DONE ingested=<int> reclaimed=<int>' line, got: {first.stdout!r}"
    ingested, reclaimed = int(m.group(1)), int(m.group(2))
    assert ingested == N_MESSAGES, f"Expected ingested={N_MESSAGES}, got {ingested}."
    assert reclaimed == N_PENDING, f"Expected reclaimed={N_PENDING}, got {reclaimed}."


def test_exactly_once_row_set(scenario):
    rows = _read_table_rows()
    assert len(rows) == N_MESSAGES, (
        f"Expected exactly {N_MESSAGES} rows (exactly-once), got {len(rows)}."
    )
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == N_MESSAGES, f"Duplicate ids detected: {len(ids) - len(set(ids))} dupes."
    expected_ids = {f"msg-{i:04d}" for i in range(N_MESSAGES)}
    assert set(ids) == expected_ids, "Stored id set does not match the seeded id set."


def test_vectors_and_text_correct(scenario):
    rows = _read_table_rows()
    ev = scenario["expected_vectors"]
    et = scenario["expected_texts"]
    by_id = {r["id"]: r for r in rows}
    for mid, exp_vec in ev.items():
        assert mid in by_id, f"Missing row for {mid}."
        got = np.array(by_id[mid]["vector"], dtype=np.float32)
        assert got.shape == (VECTOR_DIM,), f"{mid} vector has wrong shape {got.shape}."
        assert np.allclose(got, exp_vec, atol=1e-5), (
            f"{mid} vector mismatch (upsert must overwrite pre-corrupted rows)."
        )
        assert by_id[mid]["text"] == et[mid], f"{mid} text mismatch."


def test_pending_list_drained(scenario):
    client = scenario["client"]
    summary = client.xpending(STREAM_KEY, GROUP_NAME)
    pending = summary["pending"] if isinstance(summary, dict) else summary[0]
    assert pending == 0, f"Expected 0 pending entries after ingest, got {pending}."


def test_group_last_delivered_id_advanced(scenario):
    client = scenario["client"]
    groups = client.xinfo_groups(STREAM_KEY)
    target = None
    gname = GROUP_NAME.encode()
    for g in groups:
        name = g.get("name")
        if name == gname or name == GROUP_NAME:
            target = g
            break
    assert target is not None, f"Group {GROUP_NAME} not found in XINFO GROUPS."
    last_delivered = target.get("last-delivered-id")
    expected = scenario["last_entry_id"]
    if isinstance(last_delivered, bytes) and isinstance(expected, str):
        expected = expected.encode()
    if isinstance(last_delivered, str) and isinstance(expected, bytes):
        expected = expected.decode()
    assert last_delivered == expected, (
        f"Group last-delivered-id {last_delivered!r} != last seeded entry id {expected!r}."
    )


def test_rerun_is_idempotent(scenario):
    # Second run: nothing new, nothing pending -> a graceful no-op that keeps state intact.
    second = _run_consumer(CONSUMER_NAME)
    assert second.returncode == 0, f"Second consumer run failed: {second.stderr}"
    m = DONE_RE.search(second.stdout)
    assert m, f"Expected a 'DONE ...' line on rerun, got: {second.stdout!r}"
    ingested, reclaimed = int(m.group(1)), int(m.group(2))
    assert ingested == 0, f"Rerun should ingest 0 new entries, got {ingested}."
    assert reclaimed == 0, f"Rerun should reclaim 0 entries, got {reclaimed}."

    rows = _read_table_rows()
    assert len(rows) == N_MESSAGES, f"Rerun changed row count to {len(rows)} (expected {N_MESSAGES})."

    client = scenario["client"]
    summary = client.xpending(STREAM_KEY, GROUP_NAME)
    pending = summary["pending"] if isinstance(summary, dict) else summary[0]
    assert pending == 0, f"Rerun left {pending} pending entries."
