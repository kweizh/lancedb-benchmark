#!/usr/bin/env python3
"""Exactly-once Kafka -> LanceDB ingestion consumer.

Reads every document from a partitioned Kafka topic and writes one row per
document into a LanceDB table with exactly-once semantics:

  * manual offset management (auto-commit disabled; offsets are committed only
    AFTER the corresponding rows are durably written to LanceDB), and
  * an idempotent writer (LanceDB ``merge_insert`` upsert on the document ``id``
    so that replays / duplicate keys / re-consumes never create duplicate rows).

The consumer drains the currently-available messages and then exits cleanly
with status code 0 once polling yields no new records for a few consecutive
seconds.  Re-running the command is safe and idempotent.
"""

import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import lancedb
import pyarrow as pa
from confluent_kafka import (
    Consumer,
    KafkaException,
    TopicPartition,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LANCEDB_DIR = "/home/user/myproject/lancedb"
RUN_ID_FILE = "/logs/artifacts/run-id"
DEFAULT_RUN_ID = "zrlocal"
DEFAULT_BOOTSTRAP = "localhost:9092"

VECTOR_DIM = 32

# Batch size: how many records to accumulate before doing a durable write +
# offset commit.  Anything buffered is always flushed before exit.
BATCH_SIZE = 500

# Poll timeout (seconds) for a single consumer.poll() call.
POLL_TIMEOUT = 1.0

# Drain condition: exit once polling yields no new records for this many
# consecutive seconds (after the consumer group has been assigned partitions).
IDLE_SECONDS = 5.0

# Hard cap on total runtime / waiting for a group assignment, in seconds.
ASSIGNMENT_TIMEOUT = 30.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def read_run_id() -> str:
    """Read the run id from the artifacts file, falling back to the default."""
    try:
        with open(RUN_ID_FILE, "r", encoding="utf-8") as fh:
            rid = fh.read().strip()
        if rid:
            return rid
    except FileNotFoundError:
        pass
    return DEFAULT_RUN_ID


def compute_vector(text: str) -> List[float]:
    """Deterministic, local 32-dim embedding of ``text``.

    Start from a zero vector of length 32; for every whitespace-separated token
    of the lowercased ``text`` add 1.0 to ``vector[sha1(token) % 32]``; finally
    L2-normalize (leaving it all zeros if the norm is 0).
    """
    vec = [0.0] * VECTOR_DIM
    for token in text.lower().split():
        bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % VECTOR_DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def get_schema() -> pa.Schema:
    """The exact LanceDB schema for the documents table."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("ts", pa.int64()),
            pa.field("partition", pa.int64()),
            pa.field("offset", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def open_table(db: lancedb.DBConnection, table_name: str):
    """Open the LanceDB table, creating it with the right schema if absent."""
    if table_name in db.table_names():
        return db.open_table(table_name)
    return db.create_table(table_name, schema=get_schema())


def build_arrow_table(rows: List[dict]) -> pa.Table:
    """Build a pyarrow Table with the fixed-size-list vector column."""
    vector_type = pa.list_(pa.float32(), VECTOR_DIM)
    return pa.table(
        {
            "id": pa.array([r["id"] for r in rows], pa.string()),
            "text": pa.array([r["text"] for r in rows], pa.string()),
            "source": pa.array([r["source"] for r in rows], pa.string()),
            "ts": pa.array([r["ts"] for r in rows], pa.int64()),
            "partition": pa.array([r["partition"] for r in rows], pa.int64()),
            "offset": pa.array([r["offset"] for r in rows], pa.int64()),
            "vector": pa.array(
                [r["vector"] for r in rows], vector_type
            ),
        }
    )


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Core flush: durable write + post-write offset commit
# --------------------------------------------------------------------------- #

def flush_batch(
    consumer: Consumer,
    table,
    buffer: List[dict],
    max_offsets: Dict[int, int],
    topic: str,
) -> None:
    """Durably write the buffered rows to LanceDB (idempotent upsert) and then
    commit the corresponding Kafka offsets.

    Order matters: write first, commit offsets only after the write succeeds.
    """
    if not buffer:
        # Nothing to write, but still commit any tracked offsets (none here).
        return

    # Deduplicate within this batch by `id` (the idempotency key), keeping the
    # last occurrence so a later replay/redelivery wins.  merge_insert dedupes
    # against the *existing* table rows but not within the new batch itself, so
    # we must collapse intra-batch duplicate keys here.
    deduped: Dict[str, dict] = {}
    for row in buffer:
        deduped[row["id"]] = row
    rows = build_arrow_table(list(deduped.values()))
    # Idempotent upsert on `id`: a single row per id is kept regardless of
    # replays, duplicate keys, or a full re-consume after an offset reset.
    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(rows)
    )

    # Build the list of offsets to commit (offset = next offset to consume).
    offsets = [
        TopicPartition(topic, p, o + 1) for p, o in max_offsets.items()
    ]
    if offsets:
        # Synchronous commit so we only proceed once the broker has recorded
        # the committed offsets.
        consumer.commit(offsets=offsets, asynchronous=False)

    log(
        f"flushed {len(buffer)} record(s) to LanceDB and committed "
        f"{len(offsets)} partition offset(s): "
        f"{ {p: o for p, o in max_offsets.items()} }"
    )

    buffer.clear()
    max_offsets.clear()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    run_id = read_run_id()
    topic = f"ingest-docs-{run_id}"
    group = f"ingest-group-{run_id}"
    table_name = f"documents_{run_id}"
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS") or DEFAULT_BOOTSTRAP

    log(f"run_id={run_id}")
    log(f"topic={topic} group={group} table={table_name}")
    log(f"bootstrap={bootstrap}")
    log(f"lancedb_dir={LANCEDB_DIR}")

    # ---- LanceDB ---------------------------------------------------------- #
    db = lancedb.connect(LANCEDB_DIR)
    table = open_table(db, table_name)
    log(f"opened LanceDB table '{table_name}' (rows={table.count_rows()})")

    # ---- Kafka consumer --------------------------------------------------- #
    conf = {
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "enable.auto.commit": False,  # manual offset management
        "auto.offset.reset": "earliest",  # re-consume from start after reset
        "session.timeout.ms": 10000,
        "heartbeat.interval.ms": 3000,
        "max.poll.interval.ms": 300000,
        "partition.assignment.strategy": "cooperative-sticky",
        "enable.partition.eof": True,
        # Quiet down the librdkafka logger a bit.
        "log.queue": True,
    }

    assigned_partitions: set = set()

    def on_assign(consumer, parts):
        for p in parts:
            assigned_partitions.add(p.partition)
        log(f"assigned partitions: {sorted(assigned_partitions)}")

    def on_revoke(consumer, parts):
        for p in parts:
            assigned_partitions.discard(p.partition)
        log(f"revoked partitions: {[p.partition for p in parts]}")

    consumer = Consumer(conf)

    try:
        consumer.subscribe([topic], on_assign=on_assign, on_revoke=on_revoke)
        log(f"subscribed to topic '{topic}'")

        buffer: List[dict] = []
        max_offsets: Dict[int, int] = {}
        total_consumed = 0

        start_time = time.time()
        last_progress = time.time()
        assigned = False

        while True:
            msg = consumer.poll(timeout=POLL_TIMEOUT)

            if msg is None:
                # No message available right now.
                pass
            elif msg.error() is not None:
                # Partition EOF is normal (end of currently-available records
                # for a partition); treat as an idle tick, not a fatal error.
                err = msg.error()
                if err.code() == err._PARTITION_EOF:
                    pass
                else:
                    log(f"kafka error: {err}")
            else:
                # A real record.
                key = msg.key()
                if key is not None:
                    key = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else key
                value_raw = msg.value()
                try:
                    doc = json.loads(value_raw.decode("utf-8"))
                except Exception as exc:
                    log(f"failed to parse message (key={key!r}): {exc}")
                    raise

                doc_id = doc.get("id", key)
                row = {
                    "id": doc_id,
                    "text": doc["text"],
                    "source": doc["source"],
                    "ts": int(doc["ts"]),
                    "partition": int(msg.partition()),
                    "offset": int(msg.offset()),
                    "vector": compute_vector(doc["text"]),
                }
                buffer.append(row)
                p = int(msg.partition())
                # Track the highest offset seen per partition in this batch.
                if p not in max_offsets or row["offset"] > max_offsets[p]:
                    max_offsets[p] = row["offset"]
                total_consumed += 1
                last_progress = time.time()

                # Durable write + commit once the batch is full.
                if len(buffer) >= BATCH_SIZE:
                    flush_batch(consumer, table, buffer, max_offsets, topic)

            # ---- Drain / exit logic --------------------------------------- #
            now = time.time()

            # Consider the group "assigned" once we know our partition set.
            if assigned_partitions:
                assigned = True

            elapsed_since_progress = now - last_progress
            elapsed_total = now - start_time

            if assigned and elapsed_since_progress >= IDLE_SECONDS:
                # No new records for a few consecutive seconds: drain complete.
                log(
                    f"no new records for {elapsed_since_progress:.1f}s; draining"
                )
                break

            if not assigned and elapsed_total >= ASSIGNMENT_TIMEOUT:
                log(
                    "no partition assignment received within timeout; exiting"
                )
                break

        # ---- Final flush of any remaining buffered records ------------------ #
        if buffer:
            flush_batch(consumer, table, buffer, max_offsets, topic)

        log(f"done. consumed={total_consumed} table_rows={table.count_rows()}")
        return 0

    except KeyboardInterrupt:
        log("interrupted")
        return 130
    finally:
        consumer.close()


if __name__ == "__main__":
    sys.exit(main())