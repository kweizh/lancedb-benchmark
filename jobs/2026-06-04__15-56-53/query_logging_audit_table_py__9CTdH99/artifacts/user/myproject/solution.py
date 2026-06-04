"""
solution.py – LoggedSearcher
Wraps a LanceDB articles table with per-query audit logging.
"""

from __future__ import annotations

import datetime
import time
from typing import List

import lancedb
import pyarrow as pa


# ---------------------------------------------------------------------------
# Schema for the query_logs table
# ---------------------------------------------------------------------------
_LOGS_SCHEMA = pa.schema(
    [
        pa.field("query_id",   pa.string()),
        pa.field("user_id",    pa.string()),
        pa.field("query_text", pa.string()),
        pa.field("ts",         pa.timestamp("us", tz="UTC")),
        pa.field("latency_ms", pa.float64()),
        pa.field("hit_count",  pa.int64()),
        pa.field("top_ids",    pa.list_(pa.int64())),
    ]
)


class LoggedSearcher:
    """
    Wraps ``lancedb.Table.search`` and persists one audit row to a
    ``query_logs`` LanceDB table for every search performed.

    Parameters
    ----------
    db_uri : str
        Path / URI of the LanceDB database directory.
    articles_table : str
        Name of the table that holds pre-computed embeddings to search.
    logs_table : str
        Name of the audit table to write query logs into.
        Created automatically on first call to :meth:`search` if absent.
    """

    def __init__(self, db_uri: str, articles_table: str, logs_table: str) -> None:
        self._db: lancedb.LanceDBConnection = lancedb.connect(db_uri)
        self._articles_table_name: str = articles_table
        self._logs_table_name: str = logs_table
        # Open the articles table eagerly so we fail fast on bad config.
        self._articles: lancedb.table.Table = self._db.open_table(articles_table)
        # The logs table is opened / created lazily on first search.
        self._logs: lancedb.table.Table | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_logs_table(self) -> lancedb.table.Table:
        """Return (and cache) the query_logs table, creating it if needed."""
        if self._logs is not None:
            return self._logs

        if self._logs_table_name in self._db.table_names():
            self._logs = self._db.open_table(self._logs_table_name)
        else:
            # Create with explicit schema so every field type is correct
            # even before the first real row arrives.
            empty = pa.table(
                {
                    "query_id":   pa.array([], type=pa.string()),
                    "user_id":    pa.array([], type=pa.string()),
                    "query_text": pa.array([], type=pa.string()),
                    "ts":         pa.array([], type=pa.timestamp("us", tz="UTC")),
                    "latency_ms": pa.array([], type=pa.float64()),
                    "hit_count":  pa.array([], type=pa.int64()),
                    "top_ids":    pa.array([], type=pa.list_(pa.int64())),
                }
            )
            self._logs = self._db.create_table(
                self._logs_table_name,
                data=empty,
                schema=_LOGS_SCHEMA,
            )

        return self._logs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector,
        top_k: int,
        query_id: str,
        user_id: str,
        query_text: str = "",
    ) -> List[dict]:
        """
        Run a vector similarity search and log the query.

        Parameters
        ----------
        query_vector :
            A sequence of 64 floats (the query embedding).
        top_k : int
            Number of nearest neighbours to return.
        query_id : str
            Caller-supplied identifier for this query.
        user_id : str
            Identifier of the user issuing the request.
        query_text : str, optional
            Human-readable query string (default ``""``).

        Returns
        -------
        list[dict]
            Top-*k* hits, each containing at least ``id`` and ``title``.
        """
        # --- timed search -------------------------------------------------
        t0 = time.perf_counter()
        hits: List[dict] = (
            self._articles.search(query_vector).limit(top_k).to_list()
        )
        latency_ms: float = (time.perf_counter() - t0) * 1_000.0

        # --- build audit row ----------------------------------------------
        ts_now = datetime.datetime.now(datetime.timezone.utc)
        top_ids: List[int] = [int(h["id"]) for h in hits]

        log_row = pa.table(
            {
                "query_id":   pa.array([query_id],             type=pa.string()),
                "user_id":    pa.array([user_id],              type=pa.string()),
                "query_text": pa.array([query_text],           type=pa.string()),
                "ts":         pa.array([ts_now],               type=pa.timestamp("us", tz="UTC")),
                "latency_ms": pa.array([latency_ms],           type=pa.float64()),
                "hit_count":  pa.array([len(hits)],            type=pa.int64()),
                "top_ids":    pa.array([top_ids],              type=pa.list_(pa.int64())),
            }
        )

        logs = self._ensure_logs_table()
        logs.add(log_row)

        return hits
