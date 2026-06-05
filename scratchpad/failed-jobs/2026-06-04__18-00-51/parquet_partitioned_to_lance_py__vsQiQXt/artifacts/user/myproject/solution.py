"""
Hive-partitioned Parquet → LanceDB ingestion + year-filtered vector search.
"""

import os
import pyarrow.dataset as ds
import lancedb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARQUET_DIR = os.path.join(_PROJECT_DIR, "parquet_dataset")
_LANCEDB_DIR = os.path.join(_PROJECT_DIR, "lancedb")
_RUN_ID = os.environ["ZEALT_RUN_ID"]
_TABLE_NAME = f"articles_{_RUN_ID}"
_EXPECTED_ROWS = 600

# ---------------------------------------------------------------------------
# Ingest (runs at import time, idempotent)
# ---------------------------------------------------------------------------

def _ingest() -> lancedb.table.LanceTable:
    """Open or create the LanceDB table, ingesting from Parquet if needed."""
    db = lancedb.connect(_LANCEDB_DIR)

    # Check whether the table already exists and is complete.
    if _TABLE_NAME in db.table_names():
        tbl = db.open_table(_TABLE_NAME)
        if tbl.count_rows() == _EXPECTED_ROWS:
            return tbl
        # Incomplete table — drop and re-create.
        db.drop_table(_TABLE_NAME)

    # Open the Hive-partitioned dataset so `year` becomes a real column.
    dataset = ds.dataset(_PARQUET_DIR, partitioning="hive")

    # Stream all batches into a new LanceDB table.
    # We pass a Python generator of RecordBatches so that we do not have to
    # materialise the entire dataset into memory at once.
    batches = dataset.to_batches()

    tbl = db.create_table(_TABLE_NAME, data=batches, schema=dataset.schema)
    return tbl


_table = _ingest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_year(vec, year: int, k: int = 5) -> list[dict]:
    """
    Vector search on `embedding` restricted server-side to the given `year`.

    Parameters
    ----------
    vec  : list or numpy array of length 24
    year : integer year to filter on
    k    : maximum number of results to return

    Returns
    -------
    List of dicts with keys ``id``, ``title``, ``year`` (ordered by
    ascending vector distance).
    """
    db = lancedb.connect(_LANCEDB_DIR)
    tbl = db.open_table(_TABLE_NAME)

    results = (
        tbl.search(vec)
           .where(f"year = {int(year)}")
           .limit(k)
           .select(["id", "title", "year"])
           .to_list()
    )

    return [
        {"id": int(row["id"]), "title": str(row["title"]), "year": int(row["year"])}
        for row in results
    ]
