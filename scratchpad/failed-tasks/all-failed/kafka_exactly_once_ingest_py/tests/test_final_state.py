import hashlib
import json
import math
import os
import subprocess
import time

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_PATH = os.path.join(PROJECT_DIR, "lancedb")
BOOTSTRAP = "127.0.0.1:9092"
NUM_PARTITIONS = 4
NUM_DOCS = 240
NUM_INSTREAM_DUP = 20
RUN_CMD = ["python3", "run.py"]
RUN_TIMEOUT = 180


def _run_id():
    try:
        with open("/logs/artifacts/run-id") as f:
            rid = f.read().strip()
        if rid:
            return rid
    except OSError:
        pass
    return "zrlocal"


RUN_ID = _run_id()
TOPIC = f"ingest-docs-{RUN_ID}"
GROUP = f"ingest-group-{RUN_ID}"
TABLE = f"documents_{RUN_ID}"


def doc_id(n):
    return f"doc-{n:04d}"


def doc_text(n):
    return f"document {n:04d} about topic {n % 7} unique token id{n:04d} alpha bravo charlie"


def doc_payload(n):
    return {
        "id": doc_id(n),
        "text": doc_text(n),
        "source": f"feed-{n % 4}",
        "ts": 1700000000 + n,
    }


def embed(text):
    v = [0.0] * 32
    for tok in text.lower().split():
        b = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16) % 32
        v[b] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0:
        v = [x / norm for x in v]
    return v


def _wait_broker():
    import socket

    for _ in range(60):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", 9092))
            s.close()
            return
        except OSError:
            try:
                s.close()
            except OSError:
                pass
            time.sleep(1)
    raise RuntimeError("Redpanda broker not reachable on 127.0.0.1:9092")


def _ensure_fresh_topic(admin):
    from confluent_kafka.admin import NewTopic

    md = admin.list_topics(timeout=15)
    if TOPIC in md.topics:
        fs = admin.delete_topics([TOPIC], operation_timeout=30)
        try:
            fs[TOPIC].result(timeout=60)
        except Exception:
            pass
        for _ in range(60):
            md = admin.list_topics(timeout=15)
            if TOPIC not in md.topics:
                break
            time.sleep(1)
    fs = admin.create_topics(
        [NewTopic(TOPIC, num_partitions=NUM_PARTITIONS, replication_factor=1)]
    )
    fs[TOPIC].result(timeout=60)
    for _ in range(60):
        md = admin.list_topics(timeout=15)
        t = md.topics.get(TOPIC)
        if t is not None and len(t.partitions) == NUM_PARTITIONS:
            return
        time.sleep(1)
    raise RuntimeError(f"Topic {TOPIC} not ready with {NUM_PARTITIONS} partitions")


def _produce(payloads):
    from confluent_kafka import Producer

    p = Producer({"bootstrap.servers": BOOTSTRAP})
    for pl in payloads:
        p.produce(
            TOPIC,
            key=pl["id"].encode("utf-8"),
            value=json.dumps(pl).encode("utf-8"),
        )
    p.flush(60)


def _drop_table():
    import lancedb

    try:
        db = lancedb.connect(DB_PATH)
        if TABLE in db.table_names():
            db.drop_table(TABLE)
    except Exception:
        pass


def _read_rows():
    import lancedb

    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE)
    df = tbl.to_pandas()
    return tbl, df


def _run_candidate():
    proc = subprocess.run(
        RUN_CMD,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )
    print("=== candidate stdout ===")
    print(proc.stdout)
    print("=== candidate stderr ===")
    print(proc.stderr)
    return proc


def _committed_offsets(admin):
    from confluent_kafka.admin import ConsumerGroupTopicPartitions

    fs = admin.list_consumer_group_offsets([ConsumerGroupTopicPartitions(GROUP)])
    res = fs[GROUP].result(timeout=30)
    out = {}
    for tp in res.topic_partitions:
        out[tp.partition] = tp.offset
    return out


def _high_watermarks():
    from confluent_kafka import Consumer, TopicPartition

    c = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"verifier-watermark-{RUN_ID}",
            "enable.auto.commit": False,
        }
    )
    out = {}
    try:
        for part in range(NUM_PARTITIONS):
            _low, high = c.get_watermark_offsets(
                TopicPartition(TOPIC, part), timeout=15, cached=False
            )
            out[part] = high
    finally:
        c.close()
    return out


def _reset_offsets_to_zero(admin):
    from confluent_kafka import TopicPartition
    from confluent_kafka.admin import ConsumerGroupTopicPartitions

    parts = [TopicPartition(TOPIC, part, 0) for part in range(NUM_PARTITIONS)]
    fs = admin.alter_consumer_group_offsets(
        [ConsumerGroupTopicPartitions(GROUP, parts)]
    )
    fs[GROUP].result(timeout=30)


@pytest.fixture(scope="session")
def pipeline():
    from confluent_kafka.admin import AdminClient

    _wait_broker()
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})

    _drop_table()
    _ensure_fresh_topic(admin)

    # Phase A: 240 distinct docs + 20 in-stream duplicates (ids doc-0000..doc-0019).
    phase_a = [doc_payload(n) for n in range(NUM_DOCS)]
    phase_a += [doc_payload(n) for n in range(NUM_INSTREAM_DUP)]
    _produce(phase_a)

    snap = {}

    # Run 1: normal ingestion.
    snap["run1"] = _run_candidate()
    _tbl1, df1 = _read_rows()
    snap["df1"] = df1
    snap["committed_after_run1"] = _committed_offsets(admin)
    snap["watermarks_after_run1"] = _high_watermarks()

    # Phase B1: replay the 240 distinct docs again (appended), then re-run.
    _produce([doc_payload(n) for n in range(NUM_DOCS)])
    snap["run2"] = _run_candidate()
    _tbl2, df2 = _read_rows()
    snap["df2"] = df2

    # Phase B2: simulate a crash where the durable write happened but the offset
    # commit was lost -> reset committed offsets to 0 and force a full re-consume.
    _reset_offsets_to_zero(admin)
    snap["run3"] = _run_candidate()
    tbl3, df3 = _read_rows()
    snap["df3"] = df3

    # Vector search sanity on the final table.
    q = embed(doc_text(100))
    snap["search"] = tbl3.search(q).limit(3).to_list()

    return snap


def test_run1_exits_cleanly(pipeline):
    proc = pipeline["run1"]
    assert proc.returncode == 0, (
        f"`python3 run.py` (run 1) exited with {proc.returncode}. stderr:\n{proc.stderr}"
    )


def test_row_count_and_unique_ids_after_run1(pipeline):
    df = pipeline["df1"]
    assert len(df) == NUM_DOCS, (
        f"Expected exactly {NUM_DOCS} rows after ingestion, got {len(df)}."
    )
    ids = list(df["id"])
    assert len(set(ids)) == NUM_DOCS, "Duplicate `id` values found after ingestion."
    expected = {doc_id(n) for n in range(NUM_DOCS)}
    assert set(ids) == expected, (
        "Ingested id set does not match the produced documents."
    )


def test_schema_and_partition_awareness(pipeline):
    df = pipeline["df1"]
    for col in ["id", "text", "source", "ts", "partition", "offset", "vector"]:
        assert col in df.columns, f"Expected column `{col}` in the LanceDB table."
    # vector is 32-dim
    vec = df.iloc[0]["vector"]
    assert len(list(vec)) == 32, f"Expected 32-dim vector, got length {len(list(vec))}."
    parts = set(int(p) for p in df["partition"])
    assert parts.issubset({0, 1, 2, 3}), f"Unexpected partition values: {parts}."
    assert len(parts) >= 2, (
        f"Expected data on multiple partitions, only saw partitions {parts}."
    )


def test_offsets_committed_after_write(pipeline):
    committed = pipeline["committed_after_run1"]
    watermarks = pipeline["watermarks_after_run1"]
    for part in range(NUM_PARTITIONS):
        assert part in committed, f"No committed offset for partition {part}."
        assert committed[part] == watermarks[part], (
            f"Partition {part}: committed offset {committed[part]} != high watermark "
            f"{watermarks[part]}. Offsets must be committed after the durable write."
        )


def test_idempotent_on_appended_replay(pipeline):
    proc = pipeline["run2"]
    assert proc.returncode == 0, (
        f"`python3 run.py` (run 2) exited with {proc.returncode}. stderr:\n{proc.stderr}"
    )
    df = pipeline["df2"]
    assert len(df) == NUM_DOCS, (
        f"Row count changed after replaying already-processed records: expected "
        f"{NUM_DOCS}, got {len(df)}."
    )
    assert len(set(df["id"])) == NUM_DOCS, "Duplicate ids after appended replay."


def test_no_duplication_after_crash_replay(pipeline):
    proc = pipeline["run3"]
    assert proc.returncode == 0, (
        f"`python3 run.py` (run 3, after offset reset) exited with {proc.returncode}. "
        f"stderr:\n{proc.stderr}"
    )
    df = pipeline["df3"]
    assert len(df) == NUM_DOCS, (
        f"Re-consuming from offset 0 created duplicates: expected {NUM_DOCS} rows, "
        f"got {len(df)}."
    )
    ids = list(df["id"])
    assert len(set(ids)) == NUM_DOCS, "Duplicate ids after simulated crash/replay."
    assert set(ids) == {doc_id(n) for n in range(NUM_DOCS)}, (
        "Id set changed after crash/replay."
    )


def test_vector_search_returns_target(pipeline):
    results = pipeline["search"]
    ids = [r["id"] for r in results]
    assert doc_id(100) in ids, (
        f"Vector search for doc-0100's embedding did not return it. Got {ids}."
    )
    target = next(r for r in results if r["id"] == doc_id(100))
    assert target["_distance"] < 1e-4, (
        f"Expected ~0 distance for the exact embedding, got {target['_distance']}."
    )
