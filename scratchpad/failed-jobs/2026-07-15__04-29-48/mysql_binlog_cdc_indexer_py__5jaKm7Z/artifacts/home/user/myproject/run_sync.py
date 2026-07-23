#!/usr/bin/env python3
"""MySQL binlog CDC indexer into LanceDB.

Reads MySQL row-based binlog events (INSERT/UPDATE/DELETE) for the table
``appdb.documents`` and replicates them into a LanceDB table named
``documents_index`` so that the LanceDB table always mirrors the current
contents of the MySQL source table.

The command is non-blocking: it drains all binlog events that are currently
available since the last checkpoint, applies them to LanceDB, persists the new
binlog coordinate to a checkpoint file, prints a one-line JSON summary to
stdout, and exits.

Usage:
    python3 run_sync.py
"""

import hashlib
import json
import os
import sys

import pyarrow as pa
import lancedb

import pymysql
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    WriteRowsEvent,
    UpdateRowsEvent,
    DeleteRowsEvent,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")
TABLE_NAME = "documents_index"
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoint.json")

MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "cdc")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "cdcpass")

SOURCE_SCHEMA = "appdb"
SOURCE_TABLE = "documents"

# The replication slave server id must differ from the MySQL server's own id
# (the server uses id 1).
REPLICA_SERVER_ID = 2

VECTOR_DIM = 32


# ---------------------------------------------------------------------------
# LanceDB schema
# ---------------------------------------------------------------------------
def lance_schema() -> pa.Schema:
    """Return the Arrow schema required for the documents_index table."""
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("body", pa.string()),
            pa.field("category", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


# ---------------------------------------------------------------------------
# Deterministic embedding
# ---------------------------------------------------------------------------
def compute_vector(title, body) -> list:
    """Deterministic 32-dimensional embedding of ``title + " " + body``.

    For each whitespace-separated token of ``text.lower()`` the md5 hash of
    the token is bucketed (mod 32) and 1.0 is added to that bucket.  The
    resulting vector is then L2-normalized.  A row whose text has no tokens
    keeps the all-zeros vector.
    """
    title = "" if title is None else str(title)
    body = "" if body is None else str(body)
    text = title + " " + body
    vec = [0.0] * VECTOR_DIM
    for token in text.lower().split():
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % VECTOR_DIM
        vec[bucket] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def row_to_record(values: dict) -> dict:
    """Convert a binlog row dict into a record dict for LanceDB."""
    return {
        "id": int(values["id"]),
        "title": values.get("title"),
        "body": values.get("body"),
        "category": values.get("category"),
        "price": float(values["price"]) if values.get("price") is not None else None,
        "vector": compute_vector(values.get("title"), values.get("body")),
    }


def records_to_arrow(records: list) -> pa.Table:
    """Build an Arrow table (with the LanceDB schema) from record dicts."""
    schema = lance_schema()
    arrays = []
    for field in schema:
        col_values = [rec[field.name] for rec in records]
        arrays.append(pa.array(col_values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict | None:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as fh:
                cp = json.load(fh)
            if cp.get("log_file") and cp.get("log_pos") is not None:
                return cp
        except (ValueError, OSError):
            pass
    return None


def save_checkpoint(log_file: str, log_pos: int) -> None:
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump({"log_file": log_file, "log_pos": int(log_pos)}, fh)
    os.replace(tmp_path, CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# MySQL helpers
# ---------------------------------------------------------------------------
def mysql_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset="utf8mb4",
    )


def first_binlog_coordinate() -> tuple[str, int]:
    """Return (log_file, log_pos) of the start of the earliest binlog file."""
    conn = mysql_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW BINARY LOGS")
            rows = cur.fetchall()
            if not rows:
                raise RuntimeError("No binary log files are available on the server")
            first_file = rows[0][0]
            return first_file, 4
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LanceDB table management
# ---------------------------------------------------------------------------
def get_lance_table():
    db = lancedb.connect(LANCEDB_DIR)
    if TABLE_NAME not in db.table_names():
        db.create_table(TABLE_NAME, schema=lance_schema())
    return db.open_table(TABLE_NAME)


def upsert_records(table, records: list) -> None:
    if not records:
        return
    arrow_tbl = records_to_arrow(records)
    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(arrow_tbl)
    )


def delete_ids(table, ids: list) -> None:
    if not ids:
        return
    for rid in ids:
        table.delete(f"id = {int(rid)}")


# ---------------------------------------------------------------------------
# Main sync logic
# ---------------------------------------------------------------------------
def sync() -> dict:
    # Resolve the starting binlog coordinate.
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        log_file = checkpoint["log_file"]
        log_pos = int(checkpoint["log_pos"])
    else:
        log_file, log_pos = first_binlog_coordinate()

    table = get_lance_table()

    connection_settings = dict(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        passwd=MYSQL_PASSWORD,
        charset="utf8mb4",
    )

    stream = BinLogStreamReader(
        connection_settings=connection_settings,
        server_id=REPLICA_SERVER_ID,
        log_file=log_file,
        log_pos=log_pos,
        resume_stream=True,
        blocking=False,
        only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
        only_schemas=[SOURCE_SCHEMA],
        only_tables=[SOURCE_TABLE],
    )

    inserts = 0
    updates = 0
    deletes = 0

    try:
        for event in stream:
            # Track the coordinate after every consumed event.
            cur_log_file = stream.log_file
            cur_log_pos = stream.log_pos

            if isinstance(event, WriteRowsEvent):
                records = [row_to_record(row["values"]) for row in event.rows]
                upsert_records(table, records)
                inserts += len(records)
            elif isinstance(event, UpdateRowsEvent):
                records = [row_to_record(row["after_values"]) for row in event.rows]
                upsert_records(table, records)
                updates += len(records)
            elif isinstance(event, DeleteRowsEvent):
                ids = [int(row["values"]["id"]) for row in event.rows]
                delete_ids(table, ids)
                deletes += len(ids)

            # Persist the checkpoint after each applied event so that the
            # indexer is restartable and never re-applies consumed events.
            if cur_log_pos:
                save_checkpoint(cur_log_file, cur_log_pos)
                log_file = cur_log_file
                log_pos = int(cur_log_pos)
    finally:
        stream.close()

    # Final checkpoint write (covers the no-event case too).
    save_checkpoint(log_file, log_pos)

    return {
        "inserts": inserts,
        "updates": updates,
        "deletes": deletes,
        "log_file": log_file,
        "log_pos": int(log_pos),
    }


def main() -> None:
    summary = sync()
    print(json.dumps(summary))
    sys.stdout.flush()


if __name__ == "__main__":
    main()