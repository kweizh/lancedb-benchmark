"""
Serialized single-writer guard for LanceDB using PostgreSQL advisory locks.

Exposes:
    guarded_write(writer_id, rows) -> int   (blocking lock)
    try_write(writer_id, rows) -> int | None (non-blocking lock)
"""

import json
import os

import psycopg2
import pyarrow as pa
import lancedb

# ---------------------------------------------------------------------------
# Configuration – read once at import time
# ---------------------------------------------------------------------------

_PG_DSN = dict(
    host=os.environ["PGHOST"],
    port=int(os.environ["PGPORT"]),
    user=os.environ["PGUSER"],
    dbname=os.environ["PGDATABASE"],
)
_LOCK_KEY = int(os.environ["PG_LOCK_KEY"])
_LANCEDB_URI = os.environ["LANCEDB_URI"]
_RUN_ID = os.environ.get("ZEALT_RUN_ID", "")

_DATA_TABLE_NAME = f"records_{_RUN_ID}"
_AUDIT_TABLE_NAME = f"audit_{_RUN_ID}"

# ---------------------------------------------------------------------------
# PyArrow schemas
# ---------------------------------------------------------------------------

_DATA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("value", pa.int64()),
        pa.field("writer_id", pa.utf8()),
        pa.field("seq", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 8)),
    ]
)

_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("seq", pa.int64()),
        pa.field("writer_id", pa.utf8()),
        pa.field("value", pa.int64()),
        pa.field("ids_json", pa.utf8()),
    ]
)

# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------

def _get_db() -> lancedb.DBConnection:
    return lancedb.connect(_LANCEDB_URI)


def _open_or_create_table(db: lancedb.DBConnection, name: str, schema: pa.Schema):
    """Open a table if it exists, otherwise create it empty."""
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema)


def _next_seq(audit_tbl) -> int:
    """Return the next sequence number based on the current max in the audit table."""
    try:
        result = audit_tbl.search().select(["seq"]).to_arrow()
        if result.num_rows == 0:
            return 1
        max_seq = result.column("seq").to_pylist()
        return max(max_seq) + 1
    except Exception:
        return 1


def _do_write(writer_id: str, rows: list, db: lancedb.DBConnection) -> int:
    """
    Perform the actual write inside the advisory lock critical section.
    Returns the assigned sequence number.
    """
    data_tbl = _open_or_create_table(db, _DATA_TABLE_NAME, _DATA_SCHEMA)
    audit_tbl = _open_or_create_table(db, _AUDIT_TABLE_NAME, _AUDIT_SCHEMA)

    seq = _next_seq(audit_tbl)

    # Build augmented rows for the data table
    ids = [r["id"] for r in rows]
    values = [r["value"] for r in rows]
    # All rows in a single call share the same value
    shared_value = values[0] if values else 0

    stamped_rows = {
        "id": pa.array(ids, type=pa.int64()),
        "value": pa.array(values, type=pa.int64()),
        "writer_id": pa.array([writer_id] * len(rows), type=pa.utf8()),
        "seq": pa.array([seq] * len(rows), type=pa.int64()),
        "vector": pa.array(
            [r["vector"] for r in rows],
            type=pa.list_(pa.float32(), 8),
        ),
    }
    data_batch = pa.table(stamped_rows, schema=_DATA_SCHEMA)

    # Upsert into data table (merge-insert keyed on id)
    (
        data_tbl.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(data_batch)
    )

    # Append one audit record
    audit_record = pa.table(
        {
            "seq": pa.array([seq], type=pa.int64()),
            "writer_id": pa.array([writer_id], type=pa.utf8()),
            "value": pa.array([shared_value], type=pa.int64()),
            "ids_json": pa.array([json.dumps(ids)], type=pa.utf8()),
        },
        schema=_AUDIT_SCHEMA,
    )
    audit_tbl.add(audit_record)

    return seq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def guarded_write(writer_id: str, rows: list) -> int:
    """
    Acquire a blocking PostgreSQL advisory lock, perform the write, release the
    lock (always, even on exception), and return the assigned sequence number.
    """
    conn = psycopg2.connect(**_PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    db = _get_db()

    try:
        cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            seq = _do_write(writer_id, rows, db)
        finally:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    finally:
        cur.close()
        conn.close()

    return seq


def try_write(writer_id: str, rows: list):
    """
    Attempt to acquire a non-blocking PostgreSQL advisory lock.  If the lock is
    already held, return None immediately.  Otherwise perform the write and
    return the assigned sequence number.
    """
    conn = psycopg2.connect(**_PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    db = _get_db()

    try:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,))
        acquired = cur.fetchone()[0]
        if not acquired:
            return None

        try:
            seq = _do_write(writer_id, rows, db)
        finally:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    finally:
        cur.close()
        conn.close()

    return seq
