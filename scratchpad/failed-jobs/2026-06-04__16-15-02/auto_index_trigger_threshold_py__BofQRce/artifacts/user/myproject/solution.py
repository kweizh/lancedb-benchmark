"""IndexedTable wrapper for LanceDB with auto-indexing on threshold crossings."""

import json
import math
import os
import datetime

import lancedb
import pyarrow as pa


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_thresholds() -> list[int]:
    """Return the sorted list of index-build thresholds from the env or default."""
    raw = os.environ.get("INDEX_THRESHOLDS", "[256, 512, 1024]")
    thresholds = json.loads(raw)
    return sorted(thresholds)


def _get_db_uri() -> str:
    return os.environ.get("LANCEDB_URI", "/workspace/db")


def _get_vectors_table() -> str:
    return os.environ.get("VECTORS_TABLE", "vectors")


def _get_log_table() -> str:
    return os.environ.get("LOG_TABLE", "index_build_log")


# ---------------------------------------------------------------------------
# IndexedTable
# ---------------------------------------------------------------------------

class IndexedTable:
    """Wraps a LanceDB table, automatically building an IVF_PQ index whenever
    the row count crosses a configured threshold."""

    def __init__(
        self,
        db_uri: str | None = None,
        vectors_table: str | None = None,
        log_table: str | None = None,
        thresholds: list[int] | None = None,
    ):
        self._db_uri = db_uri or _get_db_uri()
        self._vectors_table_name = vectors_table or _get_vectors_table()
        self._log_table_name = log_table or _get_log_table()
        self._thresholds = thresholds or _get_thresholds()

        self._db = lancedb.connect(self._db_uri)
        self._table = self._db.open_table(self._vectors_table_name)
        self._log_table_exists = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_rows(self, rows: list[dict]) -> None:
        """Append *rows* to the vectors table and rebuild the index if any
        threshold was crossed by this append."""

        prev_count = self._table.count_rows()

        # Build a PyArrow table from the row dicts and append
        ids = [r["id"] for r in rows]
        vectors = [r["vector"] for r in rows]

        # Flatten all vectors into a single float32 array, then wrap
        # into a FixedSizeListArray to match the table schema.
        flat_values = pa.array(
            [v for row in vectors for v in row], type=pa.float32()
        )
        vector_array = pa.FixedSizeListArray.from_arrays(
            flat_values, list_size=64,
        )
        data = pa.table(
            {"id": pa.array(ids, type=pa.int64()), "vector": vector_array}
        )
        self._table.add(data)

        new_count = self._table.count_rows()

        # Check each threshold for a crossing
        for threshold in self._thresholds:
            if prev_count < threshold <= new_count:
                self._rebuild_index(new_count)

    def search(self, vec: list[float], k: int) -> list[dict]:
        """Return *k* nearest neighbours under cosine similarity."""
        results = (
            self._table.search(vec)
            .distance_type("cosine")
            .limit(k)
            .to_list()
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self, current_row_count: int) -> None:
        """Create / replace the IVF_PQ index and record the build in the
        audit log."""

        num_partitions = max(8, int(math.sqrt(current_row_count)))

        # Create (or replace) the index
        self._table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=num_partitions,
            num_sub_vectors=8,
            replace=True,
        )

        # Wait for the index build to finish so subsequent searches use it
        self._table.wait_for_index(
            ["vector_idx"],
            timeout=datetime.timedelta(seconds=300),
        )

        # Record in the audit log
        self._write_log_entry(current_row_count, num_partitions)

    def _write_log_entry(self, row_count: int, num_partitions: int) -> None:
        """Insert one row into the audit log table, creating it on first use."""
        ts = datetime.datetime.utcnow().isoformat()

        entry = pa.table(
            {
                "row_count_at_build": pa.array([row_count], type=pa.int64()),
                "num_partitions": pa.array([num_partitions], type=pa.int64()),
                "ts": pa.array([ts], type=pa.string()),
            }
        )

        if not self._log_table_exists:
            if self._log_table_name in self._db.table_names():
                # Table already exists from a prior run
                self._db.open_table(self._log_table_name).add(entry)
            else:
                self._db.create_table(self._log_table_name, data=entry)
            self._log_table_exists = True
        else:
            self._db.open_table(self._log_table_name).add(entry)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_indexed_table() -> IndexedTable:
    """Return a ready-to-use ``IndexedTable`` connected to the seeded LanceDB."""
    return IndexedTable()