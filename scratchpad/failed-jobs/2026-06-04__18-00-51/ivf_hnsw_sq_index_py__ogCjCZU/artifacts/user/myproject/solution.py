"""
IVF_HNSW_SQ vector index solution using LanceDB.

Seeds a deterministic 1024×128 float32 dataset into a LanceDB table,
builds an IVF_HNSW_SQ index with cosine distance, and exposes a
`search(query_vec, k, nprobes)` helper.
"""

import os
from datetime import timedelta

import numpy as np
import pyarrow as pa
import lancedb

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
LANCE_DB_PATH = os.environ.get("LANCE_DB_PATH", "/tmp/lancedb_default")
ZEALT_RUN_ID = os.environ.get("ZEALT_RUN_ID", "default")
TABLE_NAME = f"vectors_{ZEALT_RUN_ID}"

# ---------------------------------------------------------------------------
# Deterministic dataset
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(2026)
_VECTORS = _RNG.standard_normal((1024, 128)).astype("float32")
_IDS = list(range(1024))

# ---------------------------------------------------------------------------
# PyArrow schema
# ---------------------------------------------------------------------------
_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 128)),
    ]
)

# ---------------------------------------------------------------------------
# Build / open the table and index
# ---------------------------------------------------------------------------

def _build_table(db: lancedb.LanceDBConnection) -> lancedb.table.LanceTable:
    """Create (or re-open) the table and ensure the IVF_HNSW_SQ index exists."""

    # Check whether the table already exists
    existing = db.table_names()

    if TABLE_NAME in existing:
        tbl = db.open_table(TABLE_NAME)
    else:
        # Build PyArrow table for the seed data
        pa_table = pa.table(
            {
                "id": pa.array(_IDS, type=pa.int32()),
                "vector": pa.array(
                    [_VECTORS[i].tolist() for i in range(1024)],
                    type=pa.list_(pa.float32(), 128),
                ),
            },
            schema=_SCHEMA,
        )
        tbl = db.create_table(TABLE_NAME, data=pa_table, schema=_SCHEMA, mode="overwrite")

    # Build (or replace) the IVF_HNSW_SQ index
    tbl.create_index(
        metric="cosine",
        num_partitions=8,
        vector_column_name="vector",
        replace=True,
        index_type="IVF_HNSW_SQ",
    )

    # Block until the index is ready
    indices = tbl.list_indices()
    index_names = [idx["name"] for idx in indices] if indices else ["vector_idx"]
    tbl.wait_for_index(index_names, timeout=timedelta(seconds=120))

    return tbl


# Initialise on import
_db = lancedb.connect(LANCE_DB_PATH)
_table = _build_table(_db)


# ---------------------------------------------------------------------------
# Public search helper
# ---------------------------------------------------------------------------

def search(query_vec, k: int, nprobes: int) -> list:
    """
    Search the IVF_HNSW_SQ index.

    Parameters
    ----------
    query_vec : array-like, shape (128,)
        The query vector.
    k : int
        Number of nearest neighbours to return.
    nprobes : int
        Number of IVF partitions to probe (higher → better recall, slower).

    Returns
    -------
    list[dict]
        Rows sorted in distance-ascending order; each row contains at least
        the integer ``id`` field.
    """
    results = (
        _table.search(query_vec)
        .metric("cosine")
        .nprobes(nprobes)
        .limit(k)
        .to_list()
    )
    return results


# ---------------------------------------------------------------------------
# Allow running as a standalone script for quick sanity checks
# ---------------------------------------------------------------------------

def main():
    q = [0.0] * 128
    hits = search(q, 10, 8)
    print(f"search([0]*128, k=10, nprobes=8) → {len(hits)} results")
    for row in hits:
        print(f"  id={row['id']}")


if __name__ == "__main__":
    main()
