"""Solution module: ingests Hive-partitioned Parquet into LanceDB and exposes year-filtered vector search."""

import os
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.dataset as pds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
PARQUET_PATH = PROJECT_DIR / "parquet_dataset"
LANCEDB_PATH = PROJECT_DIR / "lancedb"
TABLE_NAME = f"articles_{os.environ['ZEALT_RUN_ID']}"
EXPECTED_ROWS = 600  # 3 years × 200 rows each


# ---------------------------------------------------------------------------
# Ingestion (runs on import)
# ---------------------------------------------------------------------------
def _ingest():
    db = lancedb.connect(str(LANCEDB_PATH))

    # Check if the table already exists with the expected content
    if TABLE_NAME in db.table_names():
        existing_table = db.open_table(TABLE_NAME)
        if existing_table.count_rows() == EXPECTED_ROWS:
            return existing_table

    # Read the Hive-partitioned Parquet dataset, materialising the year column
    ds = pds.dataset(str(PARQUET_PATH), partitioning="hive", format="parquet")
    full_table = ds.to_table()

    # Create (or overwrite) the LanceDB table with the complete dataset
    table = db.create_table(TABLE_NAME, data=full_table, mode="overwrite")
    return table


_table = _ingest()


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------
def search_year(vec, year: int, k: int = 5) -> list[dict]:
    """Return up to *k* rows matching *vec* (24-dim) restricted to *year*.

    The year filter is applied server-side via LanceDB's SQL ``where`` clause.
    """
    results = (
        _table.search(vec, vector_column_name="embedding")
        .where(f"year = {int(year)}")
        .limit(k)
        .to_list()
    )

    # Strip internal columns like _distance, keep only the requested fields
    out: list[dict] = []
    for row in results:
        out.append(
            {
                "id": int(row["id"]),
                "title": str(row["title"]),
                "year": int(row["year"]),
            }
        )
    return out