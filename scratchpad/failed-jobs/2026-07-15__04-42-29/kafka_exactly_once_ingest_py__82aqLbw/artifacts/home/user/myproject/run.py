"""
Exactly-once Kafka -> LanceDB ingestion consumer.

Design:
  - Manual offset management (enable.auto.commit=false).
  - Commit offsets only AFTER the batch is durably written to LanceDB.
  - Idempotent writes via merge_insert (upsert on `id`).
  - Drain strategy: exit cleanly after IDLE_POLLS consecutive empty polls.
"""

import hashlib
import json
import math
import os
import sys
import time
from typing import Optional

import pyarrow as pa
import lancedb
from confluent_kafka import Consumer, KafkaError, TopicPartition

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

RUN_ID_FILE = "/logs/artifacts/run-id"
try:
    with open(RUN_ID_FILE) as fh:
        RUN_ID = fh.read().strip()
except FileNotFoundError:
    RUN_ID = "zrlocal"

KAFKA_TOPIC = f"ingest-docs-{RUN_ID}"
KAFKA_GROUP = f"ingest-group-{RUN_ID}"
LANCEDB_DIR = "/home/user/myproject/lancedb"
TABLE_NAME = f"documents_{RUN_ID}"

# Drain: stop after this many consecutive polls that return 0 messages.
IDLE_POLLS_TO_EXIT = 5
POLL_TIMEOUT_S = 2.0        # seconds to wait per poll call
BATCH_SIZE = 500            # max records to accumulate before flushing

VECTOR_DIM = 32

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

ARROW_SCHEMA = pa.schema([
    pa.field("id",        pa.string()),
    pa.field("text",      pa.string()),
    pa.field("source",    pa.string()),
    pa.field("ts",        pa.int64()),
    pa.field("partition", pa.int64()),
    pa.field("offset",    pa.int64()),
    pa.field("vector",    pa.list_(pa.float32(), VECTOR_DIM)),
])

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Deterministic local bag-of-words embedding of length VECTOR_DIM.

    For each whitespace-separated token of the lowercased text:
        bucket = int(sha1(token.encode()).hexdigest(), 16) % VECTOR_DIM
        vector[bucket] += 1.0
    Then L2-normalise (leave as all-zeros if norm is zero).
    """
    vec = [0.0] * VECTOR_DIM
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % VECTOR_DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec

# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------

def open_or_create_table(db: lancedb.LanceDBConnection) -> lancedb.table.Table:
    """Open the table if it exists, otherwise create it with the correct schema."""
    existing = db.table_names()
    if TABLE_NAME in existing:
        return db.open_table(TABLE_NAME)

    # Create with an empty batch so schema is set correctly.
    empty = pa.table(
        {
            "id":        pa.array([], type=pa.string()),
            "text":      pa.array([], type=pa.string()),
            "source":    pa.array([], type=pa.string()),
            "ts":        pa.array([], type=pa.int64()),
            "partition": pa.array([], type=pa.int64()),
            "offset":    pa.array([], type=pa.int64()),
            "vector":    pa.array([], type=pa.list_(pa.float32(), VECTOR_DIM)),
        },
        schema=ARROW_SCHEMA,
    )
    return db.create_table(TABLE_NAME, data=empty, schema=ARROW_SCHEMA)


def upsert_batch(table: lancedb.table.Table, rows: list[dict]) -> None:
    """
    Idempotent batch write: merge_insert on `id`.
    Any row whose `id` already exists is updated; new ids are inserted.
    """
    if not rows:
        return

    # Build PyArrow table for the batch.
    ids        = pa.array([r["id"]        for r in rows], type=pa.string())
    texts      = pa.array([r["text"]      for r in rows], type=pa.string())
    sources    = pa.array([r["source"]    for r in rows], type=pa.string())
    tss        = pa.array([r["ts"]        for r in rows], type=pa.int64())
    partitions = pa.array([r["partition"] for r in rows], type=pa.int64())
    offsets    = pa.array([r["offset"]    for r in rows], type=pa.int64())
    vectors    = pa.array(
        [r["vector"] for r in rows],
        type=pa.list_(pa.float32(), VECTOR_DIM),
    )

    batch = pa.table(
        {
            "id":        ids,
            "text":      texts,
            "source":    sources,
            "ts":        tss,
            "partition": partitions,
            "offset":    offsets,
            "vector":    vectors,
        },
        schema=ARROW_SCHEMA,
    )

    (
        table.merge_insert("id")
             .when_matched_update_all()
             .when_not_matched_insert_all()
             .execute(batch)
    )


# ---------------------------------------------------------------------------
# Rebalance callbacks
# ---------------------------------------------------------------------------

# We need access to the consumer inside the callbacks; store it at module level.
_consumer: Optional[Consumer] = None

def on_assign(consumer, partitions):
    print(f"[rebalance] Assigned partitions: {[p.partition for p in partitions]}", flush=True)

def on_revoke(consumer, partitions):
    """
    On revoke we must commit any pending offsets that have been written to
    LanceDB but not yet committed to Kafka.  The main loop commits after every
    flush, so by the time revoke fires there should be nothing un-committed;
    but we call commit() defensively to be safe.
    """
    print(f"[rebalance] Revoking partitions: {[p.partition for p in partitions]}", flush=True)
    try:
        consumer.commit(asynchronous=False)
    except Exception as exc:
        print(f"[rebalance] commit on revoke failed (may be OK if nothing pending): {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"Run id      : {RUN_ID}", flush=True)
    print(f"Kafka topic : {KAFKA_TOPIC}", flush=True)
    print(f"Kafka group : {KAFKA_GROUP}", flush=True)
    print(f"LanceDB dir : {LANCEDB_DIR}", flush=True)
    print(f"Table name  : {TABLE_NAME}", flush=True)
    print(f"Brokers     : {BOOTSTRAP_SERVERS}", flush=True)

    # ------------------------------------------------------------------
    # Open LanceDB
    # ------------------------------------------------------------------
    db = lancedb.connect(LANCEDB_DIR)
    table = open_or_create_table(db)
    print(f"LanceDB table ready (rows={table.count_rows()})", flush=True)

    # ------------------------------------------------------------------
    # Create Kafka consumer
    # ------------------------------------------------------------------
    conf = {
        "bootstrap.servers":  BOOTSTRAP_SERVERS,
        "group.id":           KAFKA_GROUP,
        "enable.auto.commit": False,          # manual commits only
        "auto.offset.reset":  "earliest",     # start from the beginning if no committed offset
        "session.timeout.ms": 30_000,
        "max.poll.interval.ms": 300_000,
    }

    global _consumer
    _consumer = Consumer(conf)
    _consumer.subscribe(
        [KAFKA_TOPIC],
        on_assign=on_assign,
        on_revoke=on_revoke,
    )

    batch: list[dict] = []
    idle_polls = 0
    total_written = 0

    print("Starting poll loop …", flush=True)
    try:
        while True:
            msg = _consumer.poll(timeout=POLL_TIMEOUT_S)

            # ----------------------------------------------------------
            # Nothing arrived this poll cycle
            # ----------------------------------------------------------
            if msg is None:
                # Flush whatever is in the batch before counting idle polls.
                if batch:
                    _flush(table, batch, _consumer)
                    total_written += len(batch)
                    batch = []
                    idle_polls = 0          # a flush resets the idle counter
                else:
                    idle_polls += 1
                    print(
                        f"  idle poll {idle_polls}/{IDLE_POLLS_TO_EXIT} "
                        f"(total written so far: {total_written})",
                        flush=True,
                    )
                    if idle_polls >= IDLE_POLLS_TO_EXIT:
                        print("Topic appears drained – exiting.", flush=True)
                        break
                continue

            # ----------------------------------------------------------
            # Kafka error
            # ----------------------------------------------------------
            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    # End of a partition – not a fatal error.
                    print(
                        f"  EOF: partition {msg.partition()} offset {msg.offset()}",
                        flush=True,
                    )
                    continue
                # Any other error is unexpected – re-raise.
                raise KafkaError(err)

            # ----------------------------------------------------------
            # Valid message – decode and accumulate
            # ----------------------------------------------------------
            idle_polls = 0      # reset idle counter on any real message

            key_bytes = msg.key()
            val_bytes = msg.value()

            try:
                payload = json.loads(val_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(
                    f"  WARN: skipping malformed message at "
                    f"partition={msg.partition()} offset={msg.offset()}: {exc}",
                    flush=True,
                )
                continue

            doc_id  = payload.get("id", key_bytes.decode("utf-8") if key_bytes else "")
            text    = payload.get("text", "")
            source  = payload.get("source", "")
            ts      = int(payload.get("ts", 0))
            vector  = embed_text(text)

            batch.append({
                "id":        doc_id,
                "text":      text,
                "source":    source,
                "ts":        ts,
                "partition": msg.partition(),
                "offset":    msg.offset(),
                "vector":    vector,
            })

            # Flush when batch is full.
            if len(batch) >= BATCH_SIZE:
                _flush(table, batch, _consumer)
                total_written += len(batch)
                batch = []

    finally:
        _consumer.close()

    print(f"Done. Total rows upserted this run: {total_written}", flush=True)
    print(f"LanceDB table final row count     : {table.count_rows()}", flush=True)


def _flush(
    table: lancedb.table.Table,
    batch: list[dict],
    consumer: Consumer,
) -> None:
    """
    Write *batch* to LanceDB (idempotent upsert), then commit the
    corresponding Kafka offsets.  The order matters: write first, then commit.
    If we crash between the write and the commit, Kafka will redeliver the
    same records on restart, but the upsert ensures they are still idempotent.
    """
    upsert_batch(table, batch)

    # Commit offsets: for each partition, commit (max_offset + 1).
    offsets_by_partition: dict[int, int] = {}
    for row in batch:
        p = row["partition"]
        o = row["offset"]
        if p not in offsets_by_partition or o > offsets_by_partition[p]:
            offsets_by_partition[p] = o

    topic_partitions = [
        TopicPartition(KAFKA_TOPIC, p, o + 1)
        for p, o in offsets_by_partition.items()
    ]
    consumer.commit(offsets=topic_partitions, asynchronous=False)

    print(
        f"  Flushed {len(batch)} rows | "
        f"committed offsets: { {p: o+1 for p, o in offsets_by_partition.items()} }",
        flush=True,
    )


if __name__ == "__main__":
    run()
    sys.exit(0)
