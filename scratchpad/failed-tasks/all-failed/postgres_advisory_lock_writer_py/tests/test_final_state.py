import json
import os
import sys

import multiprocessing as mp

import numpy as np
import psycopg2
import pytest

sys.path.insert(0, "/home/user/myproject")

RUN_ID = os.environ["ZEALT_RUN_ID"]
LANCEDB_URI = os.environ["LANCEDB_URI"]
LOCK_KEY = int(os.environ["PG_LOCK_KEY"])
DATA_TABLE = f"records_{RUN_ID}"
AUDIT_TABLE = f"audit_{RUN_ID}"

NUM_WORKERS = 6
OPS_PER_WORKER = 20
ID_POOL = 30
ROWS_PER_OP = 3
TOTAL_OPS = NUM_WORKERS * OPS_PER_WORKER  # 120


def expected_vector(value):
    return np.random.default_rng(int(value)).standard_normal(8).astype("float32")


def op_ids(value):
    picks = np.random.default_rng(int(value) + 7_000_000).choice(ID_POOL, size=ROWS_PER_OP, replace=False)
    return [int(x) for x in picks]


def _worker(worker_index):
    # Runs in a forked child process; import the candidate module fresh here.
    import solution

    for j in range(OPS_PER_WORKER):
        value = worker_index * 100 + j
        vec = [float(x) for x in expected_vector(value)]
        ids = op_ids(value)
        rows = [{"id": i, "value": int(value), "vector": vec} for i in ids]
        solution.guarded_write(f"writer-{worker_index}", rows)


def _pg_connect():
    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        user=os.environ["PGUSER"],
        dbname=os.environ["PGDATABASE"],
    )
    conn.autocommit = True
    return conn


def _read_table(name):
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    return db.open_table(name).to_pandas()


@pytest.fixture(scope="module", autouse=True)
def run_concurrency():
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_worker, args=(w,)) for w in range(NUM_WORKERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    for p in procs:
        assert p.exitcode == 0, f"A concurrent writer process failed with exit code {p.exitcode}."
    yield


def test_01_audit_sequence_contiguous_and_all_values_present():
    audit = _read_table(AUDIT_TABLE)
    seqs = sorted(int(s) for s in audit["seq"].tolist())
    m = len(seqs)
    assert m >= TOTAL_OPS, f"Expected at least {TOTAL_OPS} audit rows, found {m}."
    assert seqs == list(range(1, m + 1)), (
        "Audit sequence must be strictly contiguous starting at 1 (no gaps, no duplicates). "
        f"Got {seqs[:10]}...{seqs[-5:]}"
    )

    values = [int(v) for v in audit["value"].tolist()]
    expected_values = {w * 100 + j for w in range(NUM_WORKERS) for j in range(OPS_PER_WORKER)}
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    for ev in expected_values:
        assert counts.get(ev, 0) == 1, (
            f"Concurrency operation value {ev} must appear exactly once in the audit table, "
            f"found {counts.get(ev, 0)}."
        )


def test_02_data_table_matches_serial_replay():
    audit = _read_table(AUDIT_TABLE)
    audit_sorted = audit.sort_values("seq")

    model = {}
    for _, entry in audit_sorted.iterrows():
        value = int(entry["value"])
        writer_id = str(entry["writer_id"])
        seq = int(entry["seq"])
        ids = json.loads(entry["ids_json"])
        vec = expected_vector(value)
        for i in ids:
            model[int(i)] = {"value": value, "writer_id": writer_id, "seq": seq, "vector": vec}

    data = _read_table(DATA_TABLE)
    actual_ids = set(int(x) for x in data["id"].tolist())
    assert actual_ids == set(model.keys()), (
        f"Data table id set does not match the replayed model. "
        f"Only in data: {sorted(actual_ids - set(model.keys()))}, "
        f"only in model: {sorted(set(model.keys()) - actual_ids)}"
    )

    for _, row in data.iterrows():
        i = int(row["id"])
        exp = model[i]
        assert int(row["value"]) == exp["value"], (
            f"id {i}: value {int(row['value'])} != expected {exp['value']} (lost/torn update)."
        )
        assert str(row["writer_id"]) == exp["writer_id"], (
            f"id {i}: writer_id {row['writer_id']} != expected {exp['writer_id']}."
        )
        assert int(row["seq"]) == exp["seq"], (
            f"id {i}: seq {int(row['seq'])} != expected {exp['seq']}."
        )
        actual_vec = np.asarray(row["vector"], dtype="float32")
        assert np.allclose(actual_vec, exp["vector"], atol=1e-5), (
            f"id {i}: stored vector does not match the value-derived vector (torn write)."
        )


def test_03_try_write_reports_contention_then_succeeds():
    import solution

    probe_value = 9001
    probe_id = 5000
    vec = [float(x) for x in expected_vector(probe_value)]
    rows = [{"id": probe_id, "value": probe_value, "vector": vec}]

    conn = _pg_connect()
    try:
        cur = conn.cursor()
        # Hold the advisory lock from the verifier's own session.
        cur.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
        cur.fetchone()

        audit_before = _read_table(AUDIT_TABLE)
        count_before = len(audit_before)

        result = solution.try_write("probe", rows)
        assert result is None, (
            "try_write must return None when the advisory lock is already held by another session."
        )

        audit_mid = _read_table(AUDIT_TABLE)
        assert len(audit_mid) == count_before, "try_write under contention must not append an audit record."
        data_mid = _read_table(DATA_TABLE)
        assert probe_id not in set(int(x) for x in data_mid["id"].tolist()), (
            "try_write under contention must not write any data rows."
        )

        # Release the lock so the next try_write can succeed.
        cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        cur.fetchone()
        cur.close()
    finally:
        conn.close()

    max_before = int(_read_table(AUDIT_TABLE)["seq"].max())
    seq = solution.try_write("probe", rows)
    assert isinstance(seq, int), "try_write must return the assigned integer sequence when the lock is free."
    assert seq == max_before + 1, (
        f"try_write should assign the next sequence {max_before + 1}, returned {seq}."
    )

    data_after = _read_table(DATA_TABLE)
    matched = data_after[data_after["id"] == probe_id]
    assert len(matched) == 1, "The successful try_write did not upsert the probe row."
    assert str(matched.iloc[0]["writer_id"]) == "probe", "Probe row has the wrong writer_id."
    assert int(_read_table(AUDIT_TABLE)["seq"].max()) == max_before + 1, "Audit max sequence did not advance by one."


def test_04_lock_released_on_error():
    import solution

    audit_before = _read_table(AUDIT_TABLE)
    count_before = len(audit_before)
    max_before = int(audit_before["seq"].max())

    bad_rows = [{"id": 6000, "value": 9002, "vector": [0.1, 0.2, 0.3]}]  # wrong vector length
    with pytest.raises(Exception):
        solution.guarded_write("boom", bad_rows)

    # The advisory lock must have been released despite the error.
    conn = _pg_connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
        got = cur.fetchone()[0]
        assert got is True, "Advisory lock was not released after guarded_write raised (lock leaked)."
        cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        cur.fetchone()
        cur.close()
    finally:
        conn.close()

    audit_after = _read_table(AUDIT_TABLE)
    assert len(audit_after) == count_before, "A failed guarded_write must not append an audit record."
    assert int(audit_after["seq"].max()) == max_before, "A failed guarded_write must not consume a sequence number."
