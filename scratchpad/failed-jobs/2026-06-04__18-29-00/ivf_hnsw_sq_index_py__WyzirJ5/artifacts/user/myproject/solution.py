"""Solution module: seeds a LanceDB table with deterministic vectors,
builds an IVF_HNSW_SQ index, and exposes a search() helper."""

import os
from datetime import timedelta

import lancedb
import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
DB_PATH = os.environ["LANCE_DB_PATH"]
RUN_ID = os.environ["ZEALT_RUN_ID"]
TABLE_NAME = f"vectors_{RUN_ID}"

# ---------------------------------------------------------------------------
# Connect / create table
# ---------------------------------------------------------------------------
db = lancedb.connect(DB_PATH)

# Generate the deterministic 128-d vectors
rng = np.random.default_rng(2026)
vectors = rng.standard_normal((1024, 128)).astype("float32")

# Build a PyArrow table with id (int64) and vector (list<float32>[128])
pa_table = pa.table(
    {
        "id": pa.array(range(1024), type=pa.int64()),
        "vector": pa.array(
            vectors.tolist(), type=pa.list_(pa.float32(), 128)
        ),
    }
)

# Create or overwrite the table
table = db.create_table(TABLE_NAME, pa_table, mode="overwrite")

# ---------------------------------------------------------------------------
# Create IVF_HNSW_SQ index
# ---------------------------------------------------------------------------
table.create_index(
    metric="cosine",
    num_partitions=8,
    vector_column_name="vector",
    replace=True,
    index_type="IVF_HNSW_SQ",
)

# Block until the index is ready
table.wait_for_index(["vector_idx"], timeout=timedelta(seconds=120))


# ---------------------------------------------------------------------------
# Public search helper
# ---------------------------------------------------------------------------
def search(query_vec, k, nprobes):
    """Run a cosine vector search on the indexed table.

    Parameters
    ----------
    query_vec : list[float] | array-like
        The 128-dimensional query vector.
    k : int
        Number of nearest neighbours to return.
    nprobes : int
        Number of IVF partitions to probe during search.

    Returns
    -------
    list[dict]
        Rows sorted by ascending distance, each containing at least ``id``.
    """
    results = (
        table.search(query_vec)
        .nprobes(nprobes)
        .limit(k)
        .to_list()
    )
    return results