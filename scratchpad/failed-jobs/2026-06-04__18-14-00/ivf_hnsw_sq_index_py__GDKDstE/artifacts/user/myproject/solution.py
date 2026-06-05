"""
LanceDB IVF_HNSW_SQ vector index solution.

Seeds 1024 deterministic 128-d float32 vectors into a LanceDB table,
builds an IVF_HNSW_SQ index with cosine distance, and exposes a
search() helper that accepts a nprobes parameter.
"""
import os
from datetime import timedelta

import numpy as np
import pyarrow as pa
import lancedb

# ── configuration ────────────────────────────────────────────────────────────
_DB_PATH = os.environ["LANCE_DB_PATH"]
_RUN_ID  = os.environ["ZEALT_RUN_ID"]
_TABLE_NAME = f"vectors_{_RUN_ID}"
_DIM = 128
_N   = 1024
_NUM_PARTITIONS = 8
_INDEX_NAME = "vector_idx"

# ── deterministic dataset ────────────────────────────────────────────────────
_rng     = np.random.default_rng(2026)
_vectors = _rng.standard_normal((_N, _DIM)).astype("float32")
_ids     = list(range(_N))

# ── PyArrow schema ───────────────────────────────────────────────────────────
_SCHEMA = pa.schema([
    pa.field("id",     pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), _DIM)),
])

# ── connect & seed ───────────────────────────────────────────────────────────
_db    = lancedb.connect(_DB_PATH)

# Build a PyArrow table for the seed data
_arrow_table = pa.table(
    {
        "id":     pa.array(_ids, type=pa.int64()),
        "vector": pa.array(_vectors.tolist(), type=pa.list_(pa.float32(), _DIM)),
    },
    schema=_SCHEMA,
)

# (Re-)create the table so the module is idempotent across re-runs
_table = _db.create_table(
    _TABLE_NAME,
    data=_arrow_table,
    mode="overwrite",
)

# ── build IVF_HNSW_SQ index ──────────────────────────────────────────────────
_table.create_index(
    metric="cosine",
    num_partitions=_NUM_PARTITIONS,
    vector_column_name="vector",
    replace=True,
    index_type="IVF_HNSW_SQ",
    name=_INDEX_NAME,
)

# ── wait for index to be ready ───────────────────────────────────────────────
_table.wait_for_index(
    [_INDEX_NAME],
    timeout=timedelta(seconds=120),
)


# ── public API ───────────────────────────────────────────────────────────────
def search(query_vec, k: int, nprobes: int) -> list[dict]:
    """
    Run a cosine ANN search against the IVF_HNSW_SQ index.

    Parameters
    ----------
    query_vec : array-like of shape (128,)
    k         : number of results to return
    nprobes   : number of IVF partitions to probe

    Returns
    -------
    list of dicts, sorted ascending by distance, each containing at least 'id'.
    """
    return (
        _table
        .search(query_vec)
        .metric("cosine")
        .nprobes(nprobes)
        .limit(k)
        .to_list()
    )
