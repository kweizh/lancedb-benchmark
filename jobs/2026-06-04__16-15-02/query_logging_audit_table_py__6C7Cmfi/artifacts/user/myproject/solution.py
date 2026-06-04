"""LoggedSearcher – observability wrapper around LanceDB vector search."""

from __future__ import annotations

import datetime
import time

import lancedb
import pyarrow as pa


class LoggedSearcher:
    """Wraps ``table.search(...)`` and writes one audit row per call.

    Parameters
    ----------
    db_uri : str
        Path/URI of the LanceDB database.
    articles_table : str
        Name of the table that holds the article vectors.
    logs_table : str
        Name of the table that will store query audit rows.
    """

    # PyArrow schema for the audit log table.
    _LOG_SCHEMA = pa.schema(
        [
            pa.field("query_id", pa.utf8()),
            pa.field("user_id", pa.utf8()),
            pa.field("query_text", pa.utf8()),
            pa.field("ts", pa.timestamp("us", tz="UTC")),
            pa.field("latency_ms", pa.float64()),
            pa.field("hit_count", pa.int64()),
            pa.field("top_ids", pa.list_(pa.int64())),
        ]
    )

    def __init__(self, db_uri: str, articles_table: str, logs_table: str) -> None:
        self._db_uri = db_uri
        self._articles_table_name = articles_table
        self._logs_table_name = logs_table

        self._db = lancedb.connect(db_uri)
        self._articles = self._db.open_table(articles_table)
        self._logs_initialized = False

    # ------------------------------------------------------------------
    # Lazy table creation
    # ------------------------------------------------------------------

    def _ensure_logs_table(self) -> None:
        """Create the ``query_logs`` table on first use if it doesn't exist."""
        if self._logs_initialized:
            return
        if self._logs_table_name not in self._db.table_names():
            self._db.create_table(self._logs_table_name, schema=self._LOG_SCHEMA)
        self._logs_initialized = True

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
    ) -> list[dict]:
        """Execute a vector search and log the query.

        Returns
        -------
        list[dict]
            The same list of hits that
            ``table.search(query_vector).limit(top_k).to_list()`` would return.
        """
        # ---- perform the search ----------------------------------------
        start = time.perf_counter()
        hits = self._articles.search(query_vector).limit(top_k).to_list()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # ---- extract ordered ids from results --------------------------
        top_ids = [int(hit["id"]) for hit in hits]
        hit_count = len(hits)

        # ---- build the audit row ----------------------------------------
        now = datetime.datetime.now(datetime.timezone.utc)

        audit_row = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "ts": now,
            "latency_ms": elapsed_ms,
            "hit_count": hit_count,
            "top_ids": top_ids,
        }

        # ---- persist the audit row (lazy-create table if needed) -------
        self._ensure_logs_table()
        logs_table = self._db.open_table(self._logs_table_name)
        logs_table.add([audit_row])

        return hits