"""
solution.py

Ingests a Hive-partitioned Parquet dataset into a LanceDB table and exposes a
year-filtered vector search function.
"""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import lancedb

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).parent
_PARQUET_DIR = _PROJECT_DIR / "parquet_dataset"
_LANCEDB_DIR = _PROJECT_DIR / "lancedb"
_EXPECTED_ROWS = 600

_RUN_ID = os.environ["ZEALT_RUN_ID"]
_TABLE_NAME = f"articles_{_RUN_ID}"

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _build_table(db: lancedb.LanceDBConnection) -> lancedb.table.LanceTable:
    """
    Read every batch from the Hive-partitioned dataset (materialising the
    `year` partition column) and write them into a new LanceDB table.
    """
    dataset = ds.dataset(str(_PARQUET_DIR), partitioning="hive")

    # Arrow schema that LanceDB will use; embedding stored as fixed-size list
    # of float32 (matches what pyarrow infers from the Parquet files).
    schema = pa.schema([
        pa.field("id",        pa.int64()),
        pa.field("title",     pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 24)),
        pa.field("year",      pa.int32()),
    ])

    # Create (or overwrite) the table with the correct schema, then stream
    # batches in so memory usage stays bounded.
    tbl = db.create_table(_TABLE_NAME, schema=schema, mode="overwrite")

    for batch in dataset.to_batches():
        # Cast batch to the canonical schema to guarantee types align.
        batch = batch.cast(schema)
        tbl.add(batch)

    return tbl


def _get_or_create_table(db: lancedb.LanceDBConnection) -> lancedb.table.LanceTable:
    """
    Return the existing table unchanged if it already has exactly 600 rows;
    otherwise (re)create it from the source dataset.
    """
    existing_tables = db.table_names()

    if _TABLE_NAME in existing_tables:
        tbl = db.open_table(_TABLE_NAME)
        if tbl.count_rows() == _EXPECTED_ROWS:
            return tbl  # already fully ingested – nothing to do
        # Wrong row count → recreate.

    return _build_table(db)


# ---------------------------------------------------------------------------
# Module-level initialisation (runs on import)
# ---------------------------------------------------------------------------

_db  = lancedb.connect(str(_LANCEDB_DIR))
_tbl = _get_or_create_table(_db)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_year(vec, year: int, k: int = 5) -> list[dict]:
    """
    Run a vector search on `embedding` restricted to the given `year`.

    Parameters
    ----------
    vec  : list or numpy array of 24 floats
    year : integer year to filter on (server-side SQL WHERE clause)
    k    : maximum number of results to return

    Returns
    -------
    List of up to *k* plain Python dicts with keys ``id``, ``title``,
    ``year`` (and optionally ``embedding`` / ``_distance``), ordered by
    ascending vector distance.
    """
    rows = (
        _tbl.search(vec)
            .where(f"year = {int(year)}")
            .limit(k)
            .to_list()
    )

    # Return plain dicts with at least the required keys.
    result = []
    for row in rows:
        result.append({
            "id":    int(row["id"]),
            "title": str(row["title"]),
            "year":  int(row["year"]),
        })
    return result
