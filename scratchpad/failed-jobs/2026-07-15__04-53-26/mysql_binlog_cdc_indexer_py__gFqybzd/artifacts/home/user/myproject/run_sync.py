#!/usr/bin/env python3
"""
MySQL binlog CDC indexer that mirrors changes from `appdb.documents`
into a LanceDB table named `documents_index`.

Usage:
    python3 run_sync.py

Environment:
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD must be set.

Behavior:
    * Reads row events (INSERT / UPDATE / DELETE) from MySQL binlog.
    * Replicates into LanceDB using primary-key upserts and deletes.
    * Persists last consumed binlog coordinate to checkpoint.json so
      that subsequent runs resume from where this one stopped.
    * On a fresh start (no checkpoint) it begins from the first available
      binlog file at position 4 so an empty source table is captured in full.
    * Drains all currently-available binlog events, applies them, updates
      the checkpoint, prints a one-line JSON summary, and exits.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import lancedb
import pymysql
import pyarrow as pa
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/home/user/myproject")
LANCEDB_DIR = PROJECT_DIR / "lancedb"
TABLE_NAME = "documents_index"
CHECKPOINT_FILE = PROJECT_DIR / "checkpoint.json"

SOURCE_SCHEMA = "appdb"
SOURCE_TABLE = "documents"

VECTOR_DIM = 32

MYSQL_SETTINGS = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER"),
    "password": os.environ.get("MYSQL_PASSWORD"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_vector(title, body):
    """Deterministic 32-dim L2-normalized embedding of `title + ' ' + body`.

    A row whose concatenated text contains no tokens keeps the all-zeros
    vector (no normalization performed).
    """
    text = ("" if title is None else str(title)) + " " + (
        "" if body is None else str(body)
    )
    vec = [0.0] * VECTOR_DIM
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % VECTOR_DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def lance_schema():
    return pa.schema(
        [
            ("id", pa.int64()),
            ("title", pa.string()),
            ("body", pa.string()),
            ("category", pa.string()),
            ("price", pa.float64()),
            ("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def build_arrow_batch(rows):
    """Build an Arrow RecordBatch from a list of plain row dicts."""
    ids = pa.array([r["id"] for r in rows], type=pa.int64())
    titles = pa.array([r["title"] for r in rows], type=pa.string())
    bodies = pa.array([r["body"] for r in rows], type=pa.string())
    categories = pa.array([r["category"] for r in rows], type=pa.string())
    prices = pa.array([r["price"] for r in rows], type=pa.float64())
    vectors = pa.array(
        [r["vector"] for r in rows],
        type=pa.list_(pa.float32(), VECTOR_DIM),
    )
    return pa.record_batch(
        [ids, titles, bodies, categories, prices, vectors],
        schema=lance_schema(),
    )


def row_to_upsert(values):
    """Convert a row's values dict into a plain dict ready for LanceDB."""
    title = values.get("title")
    body = values.get("body")
    price = values.get("price")
    return {
        "id": int(values["id"]),
        "title": title,
        "body": body,
        "category": values.get("category"),
        "price": float(price) if price is not None else None,
        "vector": compute_vector(title, body),
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint():
    if CHECKPOINT_FILE.is_file():
        try:
            with CHECKPOINT_FILE.open("r") as f:
                data = json.load(f)
            log_file = data.get("log_file")
            log_pos = data.get("log_pos")
            if log_file and isinstance(log_pos, int) and log_pos >= 4:
                return log_file, log_pos
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not load checkpoint ({exc})", file=sys.stderr)
    return None


def save_checkpoint(log_file, log_pos):
    tmp = CHECKPOINT_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump({"log_file": log_file, "log_pos": int(log_pos)}, f)
    os.replace(tmp, CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Starting coordinate
# ---------------------------------------------------------------------------


def first_binlog_coordinate():
    """Return (log_file, 4) for the earliest binlog file available."""
    conn = pymysql.connect(**MYSQL_SETTINGS)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW BINARY LOGS")
            rows = cur.fetchall()
            if not rows:
                raise RuntimeError("No binlog files available on the MySQL server")
            return rows[0][0], 4
    finally:
        conn.close()


def resolve_start_coordinate():
    """Resume from the checkpoint if present, else from the earliest binlog."""
    ckpt = load_checkpoint()
    if ckpt is not None:
        return ckpt[0], ckpt[1], True
    first_file, pos = first_binlog_coordinate()
    return first_file, pos, False


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------


def open_table(db):
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=lance_schema())


def apply_upserts(db, rows):
    if not rows:
        return
    batch = build_arrow_batch(rows)
    if TABLE_NAME not in db.table_names():
        db.create_table(TABLE_NAME, batch)
        return
    table = db.open_table(TABLE_NAME)
    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(batch)
    )


def apply_deletes(db, ids):
    if not ids:
        return
    if TABLE_NAME not in db.table_names():
        return
    table = db.open_table(TABLE_NAME)
    # Use unquoted integer literals since `id` is int64.
    values = ",".join(str(int(i)) for i in ids)
    table.delete(f"id IN ({values})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not MYSQL_SETTINGS["user"] or MYSQL_SETTINGS["password"] is None:
        print("MYSQL_USER / MYSQL_PASSWORD must be set", file=sys.stderr)
        return 2

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)

    start_file, start_pos, resuming = resolve_start_coordinate()

    db = lancedb.connect(str(LANCEDB_DIR))
    # Ensure the table exists before we start streaming so that the first
    # batch insert (when there is no checkpoint yet) works smoothly.
    open_table(db)

    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS,
        server_id=42,
        log_file=start_file,
        log_pos=start_pos,
        resume_stream=resuming,
        blocking=False,
        only_schemas=[SOURCE_SCHEMA],
        only_tables=[SOURCE_TABLE],
    )

    pending_upserts = []
    pending_deletes = []
    inserts = 0
    updates = 0
    deletes = 0
    last_log_file = start_file
    last_log_pos = start_pos

    try:
        for event in stream:
            last_log_file = stream.log_file
            last_log_pos = stream.log_pos

            if isinstance(event, WriteRowsEvent):
                for row in event.rows:
                    pending_upserts.append(row_to_upsert(row["values"]))
                    inserts += 1
            elif isinstance(event, UpdateRowsEvent):
                for row in event.rows:
                    pending_upserts.append(row_to_upsert(row["after_values"]))
                    updates += 1
            elif isinstance(event, DeleteRowsEvent):
                for row in event.rows:
                    pending_deletes.append(int(row["values"]["id"]))
                    deletes += 1
            # All other event types are ignored for data, but we still
            # recorded the stream's current coordinate above so the
            # checkpoint advances correctly across them.
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    apply_upserts(db, pending_upserts)
    apply_deletes(db, pending_deletes)

    save_checkpoint(last_log_file, last_log_pos)

    summary = {
        "inserts": inserts,
        "updates": updates,
        "deletes": deletes,
        "log_file": last_log_file,
        "log_pos": int(last_log_pos),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())