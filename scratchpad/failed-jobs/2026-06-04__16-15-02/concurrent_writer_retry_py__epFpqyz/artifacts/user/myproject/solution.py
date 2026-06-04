"""Concurrent SafeWriter with Retry for LanceDB Upserts."""

import fcntl
import os
import random
import time

import lancedb
import pyarrow as pa


class SafeWriter:
    """Wraps LanceDB merge_insert with advisory file-locking and retry logic.

    Ensures only one writer commits at a time via an advisory file lock,
    retries on contention with exponential backoff, and logs every attempt
    to a durable ``write_attempts`` table.
    """

    ATTEMPTS_TABLE = "write_attempts"
    MAX_ATTEMPTS = 5  # 1 initial + 4 retries
    BACKOFF_DELAYS = [0.05, 0.10, 0.20, 0.40]  # seconds (approximate)
    LOCK_HOLD_SECS = 0.005  # 5 ms artificial hold to increase contention

    def __init__(self, db_uri: str, table_name: str, key: str = "id"):
        self.db_uri = db_uri
        self.table_name = table_name
        self.key = key

        self.db = lancedb.connect(db_uri)
        self.table = self.db.open_table(table_name)
        self._ensure_attempts_table()

        # Per-table advisory lock file lives alongside the database
        self._lock_path = f"/tmp/.{table_name}.lock"

    # ------------------------------------------------------------------
    # write_attempts table management
    # ------------------------------------------------------------------

    def _ensure_attempts_table(self):
        """Create the write_attempts table if it does not already exist."""
        schema = pa.schema([
            pa.field("batch_id", pa.string()),
            pa.field("attempt_num", pa.int64()),
            pa.field("success", pa.bool_()),
            pa.field("error_msg", pa.string()),
            pa.field("ts", pa.int64()),
        ])
        if self.ATTEMPTS_TABLE not in self.db.table_names():
            try:
                self.db.create_table(self.ATTEMPTS_TABLE, schema=schema)
            except Exception:
                # Another concurrent writer may have created it first
                pass
        self.attempts_table = self.db.open_table(self.ATTEMPTS_TABLE)

    def _log_attempt(self, batch_id: str, attempt_num: int,
                     success: bool, error_msg: str):
        """Append exactly one row describing this attempt."""
        row = {
            "batch_id": [batch_id],
            "attempt_num": [attempt_num],
            "success": [success],
            "error_msg": [error_msg],
            "ts": [time.time_ns()],
        }
        self.attempts_table.add(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, batch_id: str, rows: list[dict]) -> None:
        """Upsert *rows* into the target table with retry-and-backoff.

        Parameters
        ----------
        batch_id : str
            Logical identifier for this batch (used in write_attempts logging).
        rows : list[dict]
            Rows to upsert.  Each dict must contain at least ``id``, ``value``,
            and ``ts`` keys matching the target table schema.

        Returns
        -------
        None on success.

        Raises
        ------
        Exception
            The last underlying exception if all 5 attempts are exhausted.
        """
        last_exception: Exception | None = None

        for attempt_num in range(self.MAX_ATTEMPTS):
            # Exponential backoff before retries (skip on first attempt)
            if attempt_num > 0:
                delay = self.BACKOFF_DELAYS[attempt_num - 1]
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)

            # Open a fresh file descriptor per attempt so threads don't share
            lock_fd = open(self._lock_path, "w")
            try:
                # --- Attempt to acquire advisory lock (non-blocking) ---------
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    # Lock contention – treat as a recoverable conflict
                    self._log_attempt(
                        batch_id, attempt_num, False,
                        f"Lock contention: {exc}",
                    )
                    last_exception = exc
                    continue

                # --- Lock acquired: hold briefly, then merge-insert ----------
                try:
                    # Small artificial hold so concurrent writers observe contention
                    time.sleep(self.LOCK_HOLD_SECS)

                    self.table.merge_insert(self.key) \
                        .when_matched_update_all(
                            where="source.ts > target.ts",
                        ) \
                        .when_not_matched_insert_all() \
                        .execute(rows)

                    self._log_attempt(batch_id, attempt_num, True, "")
                    return  # success
                except Exception as exc:
                    self._log_attempt(
                        batch_id, attempt_num, False, str(exc),
                    )
                    last_exception = exc
                    continue
                finally:
                    # Always release the lock before closing the fd
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                lock_fd.close()

        # All retries exhausted – propagate the last error
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("All retries exhausted without a specific exception")