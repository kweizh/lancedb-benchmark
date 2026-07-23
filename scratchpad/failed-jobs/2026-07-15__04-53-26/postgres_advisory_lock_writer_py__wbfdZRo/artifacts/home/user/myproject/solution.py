"""Serialized single-writer guard for LanceDB using PostgreSQL advisory locks.

Two public callables:

* :func:`guarded_write` -- acquires a process-global PostgreSQL advisory lock
  in **blocking** mode (``pg_advisory_lock``), performs the upsert + audit
  append atomically with respect to other writers, and releases the lock even
  if anything inside the critical section raises.
* :func:`try_write` -- identical to :func:`guarded_write` but uses the
  **non-blocking** variant ``pg_try_advisory_lock``; if the lock is already
  held by anyone else, it returns ``None`` immediately without touching
  LanceDB or consuming a sequence number.

Coordination model:
    All writers contend on a single PostgreSQL session-level advisory lock
    whose key is ``PG_LOCK_KEY``.  The advisory lock is what makes the
    "read max(seq) -> write rows -> append audit" sequence safe across
    processes.  LanceDB itself still uses optimistic concurrency for its
    commits, so the lock is the only thing preventing lost updates and
    duplicate / gap-filled sequence numbers.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

import lancedb
import psycopg2
from lancedb.table import Table


# ---------------------------------------------------------------------------
# Configuration (read once at import time from the environment)
# ---------------------------------------------------------------------------

_LOCK_KEY: int = int(os.environ["PG_LOCK_KEY"])
_LANCEDB_URI: str = os.environ["LANCEDB_URI"]
_RUN_ID: str = os.environ["ZEALT_RUN_ID"]

_DATA_TABLE_NAME: str = f"records_{_RUN_ID}"
_AUDIT_TABLE_NAME: str = f"audit_{_RUN_ID}"


def _pg_connect() -> psycopg2.extensions.connection:
    """Open a fresh psycopg2 connection using the standard ``PG*`` env vars."""
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        user=os.environ["PGUSER"],
        dbname=os.environ["PGDATABASE"],
    )


# ---------------------------------------------------------------------------
# Lazily-initialised, process-global handles to LanceDB and its tables.
#
# We cache the database handle and the open table objects so that repeated
# calls do not re-open the underlying files.  All initialisation is guarded
# by a local threading lock for the rare case of concurrent first-time
# access from multiple threads inside the same process.
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_db: Optional[lancedb.DBConnection] = None
_data_table: Optional[Table] = None
_audit_table: Optional[Table] = None


def _get_db() -> lancedb.DBConnection:
    global _db
    if _db is None:
        with _init_lock:
            if _db is None:
                _db = lancedb.connect(_LANCEDB_URI)
    return _db


def _get_data_table() -> Table:
    global _data_table
    if _data_table is None:
        with _init_lock:
            if _data_table is None:
                _data_table = _get_db().open_table(_DATA_TABLE_NAME)
    return _data_table


def _get_audit_table() -> Table:
    global _audit_table
    if _audit_table is None:
        with _init_lock:
            if _audit_table is None:
                _audit_table = _get_db().open_table(_AUDIT_TABLE_NAME)
    return _audit_table


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _next_sequence_number(audit: Table) -> int:
    """Return the next strictly-contiguous sequence number.

    The next sequence is ``max(seq) + 1`` over the audit table, or ``1`` if
    the audit table is empty.  Caller MUST hold the advisory lock.
    """
    df = audit.to_pandas()
    if df is None or len(df) == 0:
        return 1
    return int(df["seq"].max()) + 1


def _do_write(
    writer_id: str, rows: List[Dict[str, Any]], blocking: bool
) -> Optional[int]:
    """Shared body of :func:`guarded_write` and :func:`try_write`.

    Returns the assigned sequence number on success, or ``None`` if the
    non-blocking variant could not acquire the lock.
    """
    if not writer_id:
        raise ValueError("writer_id must be a non-empty string")
    if rows is None:
        rows = []

    conn = _pg_connect()
    # autocommit avoids spurious transaction-state issues with the
    # session-level advisory lock functions.
    conn.autocommit = True
    cur = conn.cursor()

    lock_held = False
    try:
        # ------------------------------------------------------------------
        # 1. Acquire the process-global advisory lock.
        # ------------------------------------------------------------------
        if blocking:
            cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
            lock_held = True
        else:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,))
            acquired = cur.fetchone()[0]
            lock_held = bool(acquired)
            if not lock_held:
                # Lock is held by someone else -- bail out without touching
                # LanceDB or consuming a sequence number.
                return None

        # ------------------------------------------------------------------
        # 2. Compute next sequence number from the audit table.
        # ------------------------------------------------------------------
        audit = _get_audit_table()
        seq = _next_sequence_number(audit)

        # ------------------------------------------------------------------
        # 3. Build the data-table payload and upsert.
        # ------------------------------------------------------------------
        ids: List[int] = []
        if rows:
            # All rows in a single call share the same `value`; the spec
            # calls this out explicitly.  We use the first row's value for
            # the audit record below.
            shared_value = int(rows[0]["value"])

            data_records: List[Dict[str, Any]] = []
            for r in rows:
                rid = int(r["id"])
                ids.append(rid)
                data_records.append(
                    {
                        "id": rid,
                        "value": int(r["value"]),
                        "writer_id": writer_id,
                        "seq": seq,
                        "vector": [float(x) for x in r["vector"]],
                    }
                )

            data_table = _get_data_table()
            (
                data_table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data_records)
            )
        else:
            shared_value = 0

        # ------------------------------------------------------------------
        # 4. Append exactly one audit record describing the operation.
        # ------------------------------------------------------------------
        audit.add(
            [
                {
                    "seq": seq,
                    "writer_id": writer_id,
                    "value": shared_value,
                    "ids_json": json.dumps(ids),
                }
            ]
        )

        return seq

    finally:
        # ------------------------------------------------------------------
        # 5. Always release the advisory lock, even on exception.
        # ------------------------------------------------------------------
        if lock_held:
            try:
                cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
            except Exception:
                # If we cannot reach PostgreSQL to unlock, the connection
                # going out of scope (or being closed below) will cause
                # PostgreSQL to release the session-level lock for us.
                pass
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def guarded_write(writer_id: str, rows: List[Dict[str, Any]]) -> int:
    """Acquire the advisory lock (blocking) and write ``rows``.

    Returns the assigned, monotonically-increasing sequence number.
    """
    return _do_write(writer_id, rows, blocking=True)


def try_write(writer_id: str, rows: List[Dict[str, Any]]) -> Optional[int]:
    """Try to acquire the advisory lock non-blockingly and write ``rows``.

    Returns the assigned sequence number on success, or ``None`` if the
    lock was held by another writer at the moment of the call.
    """
    return _do_write(writer_id, rows, blocking=False)