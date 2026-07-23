#!/usr/bin/env python3
"""Exactly-once Kafka -> LanceDB ingestion consumer.

Reads every message from the input topic ``ingest-docs-<run-id>`` and writes one
row per document into the LanceDB table ``documents_<run-id>`` under
``/home/user/myproject/lancedb``.

End-to-end exactly-once guarantees are achieved by combining:

* manual (post-write) offset commits - ``enable.auto.commit=false`` and
  ``consumer.commit(...)`` is invoked only after the corresponding rows are
  durably written to LanceDB.
* an idempotent writer - ``lancedb.Table.merge_insert("id")`` upserts each row
  on the document id, so replayed records (duplicate ids, full re-consume
  after an offset reset, etc.) never create duplicate rows.

The consumer uses ``subscribe()`` so consumer-group rebalances are handled by
the group protocol; offsets are tracked and committed per assigned partition.
For every row we persist the Kafka partition and offset that produced it.

The consumer drains currently-available messages and exits 0 cleanly once
polling yields no new records for ``DRAIN_TIMEOUT_SECONDS``. Re-running the
command is safe and idempotent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import numpy as np
import pyarrow as pa
from confluent_kafka import (
    Consumer,
    KafkaError,
    KafkaException,
    TopicPartition,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RUN_ID_PATH = "/logs/artifacts/run-id"
DEFAULT_RUN_ID = "zrlocal"
LANCEDB_DIR = "/home/user/myproject/lancedb"
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

VECTOR_DIM = 32

# Batching / draining
BATCH_FLUSH_SIZE = 500              # flush after this many accumulated messages
POLL_TIMEOUT_SECONDS = 1.0          # poll() wait per iteration
DRAIN_TIMEOUT_SECONDS = 5.0         # exit after this many seconds of empty polls
STARTUP_WAIT_SECONDS = 5.0          # also exit if no messages ever arrive


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_run_id() -> str:
    """Read run id from ``/logs/artifacts/run-id``; fall back to ``zrlocal``."""
    p = Path(RUN_ID_PATH)
    if p.is_file():
        try:
            text = p.read_text().strip()
        except OSError:
            return DEFAULT_RUN_ID
        return text or DEFAULT_RUN_ID
    return DEFAULT_RUN_ID


def compute_vector(text: str) -> np.ndarray:
    """Deterministic 32-d hash-bucket embedding, L2-normalized.

    Zero vector -> stays zero. Stored as float32.
    """
    v = np.zeros(VECTOR_DIM, dtype=np.float32)
    for tok in text.lower().split():
        digest = hashlib.sha1(tok.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % VECTOR_DIM
        v[bucket] += 1.0
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        v = v / norm
    return v.astype(np.float32, copy=False)


def build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("ts", pa.int64()),
            pa.field("partition", pa.int64()),
            pa.field("offset", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def open_table(db: lancedb.DBConnection, table_name: str, schema: pa.Schema):
    """Open the LanceDB table, creating it (with the expected schema) if needed."""
    existing = db.table_names()
    if table_name in existing:
        return db.open_table(table_name)

    # Create an empty table with the target schema so future merge_inserts
    # operate on a table with the exact column layout the writer expects.
    empty = pa.table(
        {
            "id": pa.array([], type=pa.string()),
            "text": pa.array([], type=pa.string()),
            "source": pa.array([], type=pa.string()),
            "ts": pa.array([], type=pa.int64()),
            "partition": pa.array([], type=pa.int64()),
            "offset": pa.array([], type=pa.int64()),
            "vector": pa.array([], type=pa.list_(pa.float32(), VECTOR_DIM)),
        },
        schema=schema,
    )
    return db.create_table(table_name, empty, mode="create")


# -----------------------------------------------------------------------------
# Durable write + post-write offset commit
# -----------------------------------------------------------------------------

def _build_batch_arrow(batch: List[Dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a pyarrow Table for ``merge_insert`` from accumulated messages."""
    ids = [m["id"] for m in batch]
    texts = [m["text"] for m in batch]
    sources = [m["source"] for m in batch]
    tses = [int(m["ts"]) for m in batch]
    parts = [int(m["partition"]) for m in batch]
    offs = [int(m["offset"]) for m in batch]

    # Stack per-row vectors and present them as a fixed-size-list<float32>[32].
    vecs = np.stack([compute_vector(t) for t in texts]).astype(np.float32)
    flat = pa.array(vecs.reshape(-1).tolist(), type=pa.float32())
    vector_arr = pa.FixedSizeListArray.from_arrays(flat, VECTOR_DIM)

    return pa.table(
        {
            "id": ids,
            "text": texts,
            "source": sources,
            "ts": tses,
            "partition": parts,
            "offset": offs,
            "vector": vector_arr,
        },
        schema=schema,
    )


def flush_batch(consumer: Consumer, table, batch: List[Dict[str, Any]]) -> None:
    """Durably write ``batch`` to LanceDB then commit Kafka offsets.

    Order matters: the LanceDB write is committed to disk first; only after
    that succeeds are the corresponding Kafka offsets committed. If we crash
    between the two, a replay will see the same records again - but the
    ``merge_insert`` upsert keeps the table free of duplicate rows.
    """
    if not batch:
        return

    new_tbl = _build_batch_arrow(batch, table.schema)

    # Idempotent upsert: rows with a matching ``id`` are updated in place;
    # rows with a new ``id`` are inserted. Net effect: exactly one row per id.
    table.merge_insert("id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(new_tbl)

    # Compute max offset per partition in this batch and commit (max + 1).
    max_off_per_part: Dict[int, int] = {}
    for m in batch:
        p = int(m["partition"])
        o = int(m["offset"])
        if p not in max_off_per_part or o > max_off_per_part[p]:
            max_off_per_part[p] = o

    topic = batch[0]["topic"]
    commit_offsets = [
        TopicPartition(topic, p, max_off_per_part[p] + 1)
        for p in sorted(max_off_per_part)
    ]

    try:
        consumer.commit(offsets=commit_offsets, asynchronous=False)
    except KafkaException as exc:
        # If commit fails (e.g. partition was revoked during a rebalance),
        # log and continue: the next consumer instance will replay and the
        # merge_insert upsert will keep the table idempotent.
        print(
            f"warn: commit failed for {commit_offsets}: {exc}; "
            f"will be retried on next consume",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def _record_from_message(msg) -> Optional[Dict[str, Any]]:
    """Decode one Kafka message into a record dict, or None on unrecoverable error."""
    partition = msg.partition()
    offset = msg.offset()
    topic = msg.topic()
    key_str = msg.key().decode("utf-8") if msg.key() is not None else ""

    raw = msg.value()
    if raw is None:
        # Tombstone: skip but track so the offset still advances.
        return {
            "id": key_str or f"__tombstone__p{partition}o{offset}",
            "text": "",
            "source": "",
            "ts": 0,
            "partition": partition,
            "offset": offset,
            "topic": topic,
        }

    try:
        rec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"warn: malformed JSON @ partition={partition} offset={offset}: {exc}",
            flush=True,
        )
        return None

    rec_id = rec.get("id") or key_str
    return {
        "id": str(rec_id),
        "text": str(rec.get("text", "")),
        "source": str(rec.get("source", "")),
        "ts": int(rec.get("ts", 0)),
        "partition": partition,
        "offset": offset,
        "topic": topic,
    }


def run() -> int:
    run_id = read_run_id()
    topic = f"ingest-docs-{run_id}"
    group_id = f"ingest-group-{run_id}"
    table_name = f"documents_{run_id}"

    print(f"run-id={run_id}", flush=True)
    print(f"bootstrap.servers={BOOTSTRAP_SERVERS}", flush=True)
    print(f"topic={topic} group_id={group_id} table={table_name}", flush=True)

    # LanceDB ----------------------------------------------------------------
    db = lancedb.connect(LANCEDB_DIR)
    schema = build_schema()
    table = open_table(db, table_name, schema)
    print(f"opened table '{table_name}' (rows={table.count_rows()})", flush=True)

    # Kafka consumer ---------------------------------------------------------
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "session.timeout.ms": 30000,
            "max.poll.interval.ms": 300000,
        }
    )
    consumer.subscribe([topic])

    stop_flag = {"value": False}

    def _stop(signum, frame):
        stop_flag["value"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    batch: List[Dict[str, Any]] = []
    last_msg_time = time.monotonic()
    startup = True

    try:
        while not stop_flag["value"]:
            msg = consumer.poll(POLL_TIMEOUT_SECONDS)

            if msg is None:
                # No message in this poll window.
                if batch:
                    flush_batch(consumer, table, batch)
                    batch = []
                # Drain decision: once we've seen at least one message and then
                # had no new messages for DRAIN_TIMEOUT_SECONDS, exit. We also
                # exit if no message ever arrived within STARTUP_WAIT_SECONDS
                # (e.g. topic is empty / never produced).
                now = time.monotonic()
                if startup:
                    if now - last_msg_time >= STARTUP_WAIT_SECONDS:
                        break
                else:
                    if now - last_msg_time >= DRAIN_TIMEOUT_SECONDS:
                        break
                continue

            if msg.error():
                err = msg.error()
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"kafka error: {err}", flush=True)
                continue

            rec = _record_from_message(msg)
            if rec is None:
                # Malformed payload - record a placeholder so the bad offset
                # still advances and we don't loop on the same poison pill.
                rec = {
                    "id": (msg.key().decode("utf-8") if msg.key() is not None
                           else f"__bad__p{msg.partition()}o{msg.offset()}"),
                    "text": "",
                    "source": "",
                    "ts": 0,
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "topic": msg.topic(),
                }

            batch.append(rec)
            last_msg_time = time.monotonic()
            startup = False

            if len(batch) >= BATCH_FLUSH_SIZE:
                flush_batch(consumer, table, batch)
                batch = []
    finally:
        # Drain any in-flight batch before exit (post-write commit semantics).
        if batch:
            try:
                flush_batch(consumer, table, batch)
            except Exception as exc:
                print(f"final flush failed: {exc}", flush=True)
            batch = []
        try:
            consumer.close()
        except Exception:
            pass

    final_rows = table.count_rows()
    print(f"done. table rows={final_rows}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())