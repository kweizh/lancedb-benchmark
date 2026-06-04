"""
IndexedTable – LanceDB wrapper that auto-triggers IVF_PQ index rebuilds
whenever the row count crosses a configured threshold list, and records
every rebuild in an audit log table.
"""

import datetime
import json
import math
import os

import lancedb

# ---------------------------------------------------------------------------
# Configuration (env-var overrides)
# ---------------------------------------------------------------------------
LANCEDB_URI: str = os.environ.get("LANCEDB_URI", "/workspace/db")
VECTORS_TABLE: str = os.environ.get("VECTORS_TABLE", "vectors")
LOG_TABLE: str = os.environ.get("LOG_TABLE", "index_build_log")
INDEX_THRESHOLDS: list[int] = sorted(
    json.loads(os.environ.get("INDEX_THRESHOLDS", "[256, 512, 1024]"))
)


class IndexedTable:
    """Wrapper around a LanceDB table that auto-indexes on threshold crossing."""

    def __init__(
        self,
        db: lancedb.DBConnection,
        table: lancedb.table.Table,
        thresholds: list[int] | None = None,
    ) -> None:
        self._db = db
        self._table = table
        self._thresholds: list[int] = sorted(
            thresholds if thresholds is not None else INDEX_THRESHOLDS
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_rows(self, rows: list[dict]) -> None:
        """Append *rows* to the vectors table.

        After appending, rebuild the IVF_PQ index (and write an audit log
        entry) for every configured threshold that the append crosses.
        """
        prev_count: int = self._table.count_rows()
        self._table.add(rows)
        new_count: int = self._table.count_rows()

        for threshold in self._thresholds:
            if prev_count < threshold <= new_count:
                self._rebuild_index(new_count)

    def search(self, vec: list[float], k: int) -> list[dict]:
        """Return the *k* most-cosine-similar rows to *vec*, most-similar first.

        Each element of the returned list is a dict that includes at least
        the ``id`` field of the matching row.
        """
        return (
            self._table.search(vec)
            .distance_type("cosine")
            .limit(k)
            .to_list()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self, current_row_count: int) -> None:
        """Build an IVF_PQ index on *vector* and wait for it to finish."""
        num_partitions = max(8, int(math.sqrt(current_row_count)))

        self._table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=num_partitions,
            num_sub_vectors=8,
            replace=True,
        )

        # Synchronously wait for the index to finish building.
        self._table.wait_for_index(
            ["vector_idx"],
            timeout=datetime.timedelta(seconds=300),
        )

        self._write_audit_log(current_row_count)

    def _write_audit_log(self, row_count: int) -> None:
        """Append one row to the audit log table, creating it if necessary."""
        ts = datetime.datetime.utcnow().isoformat()
        record = {
            "row_count_at_build": row_count,
            "num_partitions": max(8, int(math.sqrt(row_count))),
            "ts": ts,
        }

        existing_tables = self._db.table_names()
        if LOG_TABLE not in existing_tables:
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("row_count_at_build", pa.int64()),
                    pa.field("num_partitions", pa.int64()),
                    pa.field("ts", pa.string()),
                ]
            )
            self._db.create_table(LOG_TABLE, data=[record], schema=schema)
        else:
            self._db.open_table(LOG_TABLE).add([record])


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_indexed_table() -> IndexedTable:
    """Return a ready-to-use :class:`IndexedTable` connected to the seeded DB."""
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table(VECTORS_TABLE)
    return IndexedTable(db=db, table=table)
