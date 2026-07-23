"""Serialized single-writer guard for LanceDB using PostgreSQL advisory locks.

This module exposes two callables, ``guarded_write`` and ``try_write``, which
safely upsert rows into a shared LanceDB table while assigning a globally
contiguous, monotonically increasing sequence number.

Concurrency safety is provided by a PostgreSQL *session-level* advisory lock
(``pg_advisory_lock`` / ``pg_try_advisory_lock``).  PostgreSQL is used purely
as a cross-process coordination primitive; LanceDB itself is not modified to
rely on it for locking.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import psycopg2
import pyarrow as pa
import lancedb


# ---------------------------------------------------------------------------
# Configuration (read from the environment at call time so the module is
# importable before the harness has exported everything).
# ---------------------------------------------------------------------------

def _lock_key() -> int:
    return int(os.environ["PG_LOCK_KEY"])


def _lancedb_uri() -> str:
    return os.environ["LANCEDB_URI"]


def _run_id() -> str:
    return os.environ["ZEALT_RUN_ID"]


def _table_names() -> tuple[str, str]:
    run = _run_id()
    return f"records_{run}", f"audit_{run}"


# ---------------------------------------------------------------------------
# Schema for the data rows we hand to LanceDB's merge-insert.  Building an
# explicit pyarrow Table (instead of a list of dicts) guarantees the vector
# column is materialised as a fixed-size list<float32>[8] exactly matching the
# target table, avoiding any schema-inference / casting surprises.
# ---------------------------------------------------------------------------

_VECTOR_WIDTH = 8

_DATA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("value", pa.int64()),
        pa.field("writer_id", pa.string()),
        pa.field("seq", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), _VECTOR_WIDTH)),
    ]
)


def _build_data_table(rows: List[Dict[str, Any]], writer_id: str, seq: int) -> pa.Table:
    """Construct a pyarrow Table of stamped rows ready for merge-insert."""
    n = len(rows)
    ids = pa.array([int(r["id"]) for r in rows], type=pa.int64())
    values = pa.array([int(r["value"]) for r in rows], type=pa.int64())
    writer_ids = pa.array([writer_id] * n, type=pa.string())
    seqs = pa.array([seq] * n, type=pa.int64())

    # Flatten the per-row vectors into one contiguous float32 buffer, then wrap
    # as a fixed-size list array of width 8.
    flat: List[float] = []
    for r in rows:
        vec = r["vector"]
        if len(vec) != _VECTOR_WIDTH:
            raise ValueError(
                f"each vector must have exactly {_VECTOR_WIDTH} floats, got {len(vec)}"
            )
        flat.extend(float(x) for x in vec)
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float32()), _VECTOR_WIDTH
    )

    return pa.table(
        {
            "id": ids,
            "value": values,
            "writer_id": writer_ids,
            "seq": seqs,
            "vector": vectors,
        },
        schema=_DATA_SCHEMA,
    )


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------

def _open_tables() -> tuple[Any, Any]:
    db = lancedb.connect(_lancedb_uri())
    data_name, audit_name = _table_names()
    return db.open_table(data_name), db.open_table(audit_name)


def _next_sequence(audit_table: Any) -> int:
    """Return the next sequence number based on the audit table's max ``seq``.

    An empty audit table means the next sequence is 1.
    """
    arrow = audit_table.to_arrow()
    if arrow.num_rows == 0:
        return 1
    seqs = arrow.column("seq").to_pylist()
    if not seqs:
        return 1
    return int(max(seqs)) + 1


def _do_write(rows: List[Dict[str, Any]], writer_id: str) -> int:
    """Perform the critical-section work (assumes the advisory lock is held)."""
    if not rows:
        # Nothing to do; still we must not consume a sequence number for an
        # empty operation.  The spec implies rows is non-empty, but guard
        # against an index error defensively.
        raise ValueError("rows must contain at least one row")

    data_table, audit_table = _open_tables()

    seq = _next_sequence(audit_table)

    # Stamp + upsert the data rows keyed on ``id``.
    stamped = _build_data_table(rows, writer_id, seq)
    (
        data_table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(stamped)
    )

    # Append exactly one audit record.
    shared_value = int(rows[0]["value"])
    ids_json = json.dumps([int(r["id"]) for r in rows])
    audit_table.add(
        [
            {
                "seq": seq,
                "writer_id": writer_id,
                "value": shared_value,
                "ids_json": ids_json,
            }
        ]
    )

    return seq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def guarded_write(writer_id: str, rows: List[Dict[str, Any]]) -> int:
    """Acquire the advisory lock (blocking), then perform the write.

    Returns the assigned sequence number.  The lock is always released, even
    if the critical section raises.
    """
    key = _lock_key()
    conn = psycopg2.connect()  # uses PG* env vars via libpq defaults
    acquired = False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (key,))
            cur.fetchone()
        acquired = True
        return _do_write(rows, writer_id)
    finally:
        # Always release the lock if we hold it, even on exception.  Closing
        # the connection also releases session-level locks as a backstop.
        if acquired:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
                    cur.fetchone()
            except Exception:
                # Swallow unlock errors so the original exception propagates.
                pass
        conn.close()


def try_write(writer_id: str, rows: List[Dict[str, Any]]) -> Optional[int]:
    """Non-blocking variant of :func:`guarded_write`.

    Acquires the advisory lock with ``pg_try_advisory_lock``.  If the lock is
    currently held by anyone else, returns ``None`` immediately without writing
    or consuming a sequence number.  Otherwise performs the same work as
    ``guarded_write`` and returns the assigned sequence number.
    """
    key = _lock_key()
    conn = psycopg2.connect()
    acquired = False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
            acquired = bool(cur.fetchone()[0])
        if not acquired:
            return None
        return _do_write(rows, writer_id)
    finally:
        # Only release the lock if we actually acquired it; unlocking a key
        # we don't own would just return False (with a warning), so we skip it
        # to keep things clean.  Closing the connection also releases any
        # session-level locks as a backstop.
        if acquired:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
                    cur.fetchone()
            except Exception:
                pass
        conn.close()