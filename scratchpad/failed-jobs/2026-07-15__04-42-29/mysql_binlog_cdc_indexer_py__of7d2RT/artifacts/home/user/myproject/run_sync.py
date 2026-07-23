#!/usr/bin/env python3
"""
MySQL Binlog CDC Indexer into LanceDB
--------------------------------------
Reads MySQL row-based binlog events for appdb.documents, applies INSERT /
UPDATE / DELETE operations to a LanceDB table (documents_index), persists the
binlog checkpoint, and prints a one-line JSON summary.

Usage:
    python3 run_sync.py
"""

import hashlib
import json
import math
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
import pymysql
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import (
    DeleteRowsEvent,
    UpdateRowsEvent,
    WriteRowsEvent,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MYSQL_SETTINGS = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "cdc"),
    "passwd": os.environ.get("MYSQL_PASSWORD", "cdcpass"),
}

SOURCE_DB = "appdb"
SOURCE_TABLE = "documents"

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")
LANCEDB_TABLE = "documents_index"
CHECKPOINT_FILE = os.path.join(PROJECT_DIR, "checkpoint.json")

# The mysql-replication reader must use a server_id != the MySQL server's own
# id (the server uses id=1).
REPLICATION_SERVER_ID = 100

VECTOR_DIM = 32

# ---------------------------------------------------------------------------
# Arrow schema for the LanceDB table
# ---------------------------------------------------------------------------

LANCE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("title", pa.utf8()),
        pa.field("body", pa.utf8()),
        pa.field("category", pa.utf8()),
        pa.field("price", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ]
)


# ---------------------------------------------------------------------------
# Vector embedding
# ---------------------------------------------------------------------------

def compute_vector(title: str, body: str) -> list[float]:
    """
    Deterministic 32-dim bag-of-words embedding.

    Algorithm:
        text  = title + " " + body
        vec   = zeros(32, float)
        for token in text.lower().split():
            bucket = int(md5(token.encode("utf-8")).hexdigest(), 16) % 32
            vec[bucket] += 1.0
        L2-normalize vec (all-zeros stays all-zeros)
    Returns a Python list of 32 float32 values.
    """
    text = (title or "") + " " + (body or "")
    vec = np.zeros(VECTOR_DIM, dtype=np.float64)

    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % VECTOR_DIM
        vec[bucket] += 1.0

    norm = math.sqrt(float(np.dot(vec, vec)))
    if norm > 0.0:
        vec = vec / norm

    return vec.astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# Row → LanceDB record
# ---------------------------------------------------------------------------

def row_to_record(values: dict) -> dict:
    """Convert a binlog row-values dict to a LanceDB record."""
    return {
        "id": int(values["id"]),
        "title": values.get("title") or "",
        "body": values.get("body") or "",
        "category": values.get("category") or "",
        "price": float(values.get("price") or 0.0),
        "vector": compute_vector(
            values.get("title") or "",
            values.get("body") or "",
        ),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict | None:
    """Return {log_file, log_pos} from the checkpoint file, or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r") as fh:
            data = json.load(fh)
        if "log_file" in data and "log_pos" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def save_checkpoint(log_file: str, log_pos: int) -> None:
    """Persist the checkpoint to disk atomically."""
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"log_file": log_file, "log_pos": log_pos}, fh)
    os.replace(tmp, CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Earliest available binlog coordinate
# ---------------------------------------------------------------------------

def get_earliest_binlog(mysql_conn) -> tuple[str, int]:
    """
    Return (log_file, log_pos=4) for the oldest available binlog file.
    Position 4 skips the 4-byte magic number at the start of every binlog.
    """
    with mysql_conn.cursor() as cur:
        cur.execute("SHOW BINARY LOGS")
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No binary logs found on the MySQL server.")
    # rows are ordered oldest→newest; first column is the file name
    return rows[0][0], 4


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------

def open_or_create_lance_table(db: lancedb.DBConnection):
    """Open an existing LanceDB table or create an empty one with our schema."""
    existing = db.table_names()
    if LANCEDB_TABLE in existing:
        return db.open_table(LANCEDB_TABLE)
    # Create with an empty table but the correct schema
    empty = pa.table(
        {
            "id": pa.array([], type=pa.int64()),
            "title": pa.array([], type=pa.utf8()),
            "body": pa.array([], type=pa.utf8()),
            "category": pa.array([], type=pa.utf8()),
            "price": pa.array([], type=pa.float64()),
            "vector": pa.array(
                [], type=pa.list_(pa.float32(), VECTOR_DIM)
            ),
        }
    )
    return db.create_table(LANCEDB_TABLE, data=empty, schema=LANCE_SCHEMA)


def upsert_records(table, records: list[dict]) -> None:
    """Upsert a batch of records into the LanceDB table, keyed on 'id'."""
    if not records:
        return
    arrow_batch = pa.table(
        {
            "id": pa.array([r["id"] for r in records], type=pa.int64()),
            "title": pa.array([r["title"] for r in records], type=pa.utf8()),
            "body": pa.array([r["body"] for r in records], type=pa.utf8()),
            "category": pa.array(
                [r["category"] for r in records], type=pa.utf8()
            ),
            "price": pa.array(
                [r["price"] for r in records], type=pa.float64()
            ),
            "vector": pa.array(
                [r["vector"] for r in records],
                type=pa.list_(pa.float32(), VECTOR_DIM),
            ),
        }
    )
    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(arrow_batch)
    )


def delete_records(table, ids: list[int]) -> None:
    """Delete rows from the LanceDB table by primary key."""
    if not ids:
        return
    id_list = ", ".join(str(i) for i in ids)
    table.delete(f"id IN ({id_list})")


# ---------------------------------------------------------------------------
# Main sync logic
# ---------------------------------------------------------------------------

def run_sync() -> None:
    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.makedirs(LANCEDB_DIR, exist_ok=True)

    # --- open MySQL connection (for SHOW BINARY LOGS) ----------------------
    mysql_conn = pymysql.connect(
        host=MYSQL_SETTINGS["host"],
        port=MYSQL_SETTINGS["port"],
        user=MYSQL_SETTINGS["user"],
        password=MYSQL_SETTINGS["passwd"],
    )

    # --- resolve starting binlog coordinate --------------------------------
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        start_log_file = checkpoint["log_file"]
        start_log_pos = checkpoint["log_pos"]
        resume = True
    else:
        start_log_file, start_log_pos = get_earliest_binlog(mysql_conn)
        resume = True  # resume_stream=True + explicit file/pos = start there

    mysql_conn.close()

    # --- open LanceDB -------------------------------------------------------
    db = lancedb.connect(LANCEDB_DIR)
    table = open_or_create_lance_table(db)

    # --- open binlog stream (non-blocking) ----------------------------------
    stream = BinLogStreamReader(
        connection_settings=MYSQL_SETTINGS,
        server_id=REPLICATION_SERVER_ID,
        only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
        only_tables=[SOURCE_TABLE],
        only_schemas=[SOURCE_DB],
        log_file=start_log_file,
        log_pos=start_log_pos,
        resume_stream=resume,
        blocking=False,       # non-blocking: returns None when caught up
    )

    n_inserts = 0
    n_updates = 0
    n_deletes = 0

    # Track the coordinate; initialise to start position so that if no
    # events arrive we still save a valid checkpoint.
    last_log_file = start_log_file
    last_log_pos = start_log_pos

    for event in stream:
        # BinLogStreamReader yields None when non-blocking and end-of-log is
        # reached; stop there.
        if event is None:
            break

        current_log_file = stream.log_file
        current_log_pos = stream.log_pos

        if isinstance(event, WriteRowsEvent):
            records = [row_to_record(row["values"]) for row in event.rows]
            upsert_records(table, records)
            n_inserts += len(records)

        elif isinstance(event, UpdateRowsEvent):
            # Each row has {"before_values": {...}, "after_values": {...}}.
            # The PK (id) is immutable, so we upsert on the after image.
            records = [
                row_to_record(row["after_values"]) for row in event.rows
            ]
            upsert_records(table, records)
            n_updates += len(records)

        elif isinstance(event, DeleteRowsEvent):
            ids = [int(row["values"]["id"]) for row in event.rows]
            delete_records(table, ids)
            n_deletes += len(ids)

        last_log_file = current_log_file
        last_log_pos = current_log_pos

    stream.close()

    # --- persist checkpoint -------------------------------------------------
    save_checkpoint(last_log_file, last_log_pos)

    # --- emit JSON summary --------------------------------------------------
    summary = {
        "inserts": n_inserts,
        "updates": n_updates,
        "deletes": n_deletes,
        "log_file": last_log_file,
        "log_pos": last_log_pos,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    run_sync()
