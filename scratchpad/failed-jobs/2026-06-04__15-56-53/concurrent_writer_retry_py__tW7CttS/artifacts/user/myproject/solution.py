"""
SafeWriter: concurrent-safe LanceDB upsert wrapper with retry/backoff.
"""
from __future__ import annotations

import fcntl
import os
import random
import time
import tempfile
from typing import List, Dict, Any

import lancedb
import pyarrow as pa


# ---------------------------------------------------------------------------
# Schema for the write-attempts audit table
# ---------------------------------------------------------------------------
_ATTEMPTS_SCHEMA = pa.schema(
    [
        pa.field("batch_id", pa.string()),
        pa.field("attempt_num", pa.int64()),
        pa.field("success", pa.bool_()),
        pa.field("error_msg", pa.string()),
        pa.field("ts", pa.int64()),
    ]
)

# Retry delays in seconds (first attempt is attempt 0; delays applied before
# each subsequent attempt).
_RETRY_DELAYS = [0.05, 0.10, 0.20, 0.40]

# Artificial lock-hold time (seconds) so concurrent writers actually contend.
_LOCK_HOLD_SECS = 0.005  # 5 ms

# Jitter fraction applied to each delay (±25 %).
_JITTER_FRACTION = 0.25


def _jittered(delay: float) -> float:
    """Return *delay* ± up to JITTER_FRACTION of its value."""
    jitter = delay * _JITTER_FRACTION * (2.0 * random.random() - 1.0)
    return max(0.0, delay + jitter)


class SafeWriter:
    """
    Thread-safe LanceDB upsert wrapper with exponential-backoff retry.

    Parameters
    ----------
    db_uri:     Path (or URI) to the LanceDB database directory.
    table_name: Name of the target table to upsert into.
    key:        Column name used as the merge key (default "id").
    """

    def __init__(self, db_uri: str, table_name: str, key: str = "id") -> None:
        self._db_uri = db_uri
        self._table_name = table_name
        self._key = key

        # Each table gets its own lock file so writers to different tables
        # do not block each other.
        lock_dir = tempfile.gettempdir()
        safe_name = table_name.replace(os.sep, "_").replace("/", "_")
        self._lock_path = os.path.join(lock_dir, f"lancedb_safewriter_{safe_name}.lock")

        # Open (or create) the lock file once; the fd is kept open for the
        # lifetime of this SafeWriter instance.
        self._lock_fd = open(self._lock_path, "a")  # noqa: WPS515

        # Lazily-opened DB connection (re-opened on each call to stay fresh
        # under concurrent usage, but cached for the attempts table setup).
        self._db: lancedb.LanceDBConnection | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_db(self) -> lancedb.LanceDBConnection:
        """Return (or open) the LanceDB connection."""
        if self._db is None:
            self._db = lancedb.connect(self._db_uri)
        return self._db

    def _ensure_attempts_table(self, db: lancedb.LanceDBConnection) -> lancedb.table.LanceTable:
        """Return the write_attempts table, creating it if necessary."""
        if "write_attempts" not in db.table_names():
            empty = pa.table(
                {
                    "batch_id": pa.array([], type=pa.string()),
                    "attempt_num": pa.array([], type=pa.int64()),
                    "success": pa.array([], type=pa.bool_()),
                    "error_msg": pa.array([], type=pa.string()),
                    "ts": pa.array([], type=pa.int64()),
                }
            )
            db.create_table("write_attempts", data=empty, schema=_ATTEMPTS_SCHEMA)
        return db.open_table("write_attempts")

    def _log_attempt(
        self,
        attempts_tbl: lancedb.table.LanceTable,
        batch_id: str,
        attempt_num: int,
        success: bool,
        error_msg: str,
    ) -> None:
        """Append a single row to the write_attempts table."""
        ts_ns = time.time_ns()
        row = pa.table(
            {
                "batch_id": pa.array([batch_id], type=pa.string()),
                "attempt_num": pa.array([attempt_num], type=pa.int64()),
                "success": pa.array([success], type=pa.bool_()),
                "error_msg": pa.array([error_msg], type=pa.string()),
                "ts": pa.array([ts_ns], type=pa.int64()),
            }
        )
        attempts_tbl.add(row)

    def _acquire_lock(self) -> bool:
        """
        Try to acquire the advisory lock in non-blocking mode.

        Returns True on success, False if another writer holds the lock.
        """
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError):
            return False

    def _release_lock(self) -> None:
        """Release the advisory lock."""
        fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, batch_id: str, rows: List[Dict[str, Any]]) -> None:
        """
        Upsert *rows* into the target table.

        * Serialises concurrent writers via an advisory file lock.
        * Retries up to 4 times (5 attempts total) with exponential backoff
          (~50 ms, 100 ms, 200 ms, 400 ms) plus random jitter on lock
          contention or any transient write error.
        * Logs every attempt (success or failure) to the ``write_attempts``
          table.
        * On success returns None.
        * After all retries are exhausted raises the last exception.

        The upsert uses a timestamp-based last-writer-wins strategy: an
        existing row is only replaced when the incoming row's ``ts`` value
        is strictly greater than the stored one.

        Parameters
        ----------
        batch_id: Caller-supplied identifier for this batch (used in logs).
        rows:     List of dicts representing the rows to upsert.
        """
        db = self._get_db()
        attempts_tbl = self._ensure_attempts_table(db)
        target_tbl = db.open_table(self._table_name)

        last_exc: Exception | None = None
        # Total attempts = 1 initial + len(_RETRY_DELAYS) retries
        max_attempts = 1 + len(_RETRY_DELAYS)

        for attempt_num in range(max_attempts):
            # Apply backoff delay before each retry (not before the first try).
            if attempt_num > 0:
                delay = _jittered(_RETRY_DELAYS[attempt_num - 1])
                time.sleep(delay)

            # --- Try to acquire the per-table advisory lock ---------------
            if not self._acquire_lock():
                # Lock contention: treat as a recoverable conflict.
                err_msg = f"lock contention on attempt {attempt_num}"
                self._log_attempt(attempts_tbl, batch_id, attempt_num, False, err_msg)
                last_exc = RuntimeError(err_msg)
                continue

            # --- We hold the lock; perform the upsert ---------------------
            try:
                # Timestamp-based LWW: only overwrite if incoming ts is newer.
                (
                    target_tbl.merge_insert(self._key)
                    .when_matched_update_all(where="source.ts > target.ts")
                    .when_not_matched_insert_all()
                    .execute(rows)
                )

                # Hold the lock briefly so concurrent writers actually observe
                # contention rather than racing through too quickly.
                time.sleep(_LOCK_HOLD_SECS)

                self._log_attempt(attempts_tbl, batch_id, attempt_num, True, "")
                return  # success — exit the retry loop

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._log_attempt(
                    attempts_tbl, batch_id, attempt_num, False, str(exc)
                )

            finally:
                self._release_lock()

        # All attempts exhausted.
        raise last_exc  # type: ignore[misc]

    def __del__(self) -> None:
        """Clean up the lock file descriptor."""
        try:
            self._lock_fd.close()
        except Exception:  # noqa: BLE001
            pass
