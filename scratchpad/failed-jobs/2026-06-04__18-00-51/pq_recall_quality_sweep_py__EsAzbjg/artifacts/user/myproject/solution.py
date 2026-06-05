"""
IVF_PQ Recall-Quality Sweep with LanceDB.

Exposes a single public function:
    sweep() -> dict[int, float]

Keys are num_sub_vectors values (4, 8, 16); values are mean recall@10
against brute-force ground truth computed over the 30 fixed query vectors.
"""

from __future__ import annotations

import os
import datetime
from typing import Dict

import numpy as np
import pyarrow as pa
import lancedb

# ── paths ──────────────────────────────────────────────────────────────────────
_DB_PATH = "/home/user/myproject/lancedb_data/"
_DATA_PATH = "/app/fixtures/data.npy"
_QUERIES_PATH = "/app/fixtures/queries.npy"

# ── sweep configuration ────────────────────────────────────────────────────────
_M_VALUES = [4, 8, 16]
_NUM_PARTITIONS = 16
_NPROBES = 16
_TOP_K = 10
_INDEX_TIMEOUT = datetime.timedelta(seconds=120)


# ── brute-force ground truth ───────────────────────────────────────────────────

def _brute_force_top_k(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Return (num_queries, k) array of row indices for nearest neighbours.

    Uses squared L2 distance (equivalent to L2 for ranking purposes).
    """
    # data:    (N, D)
    # queries: (Q, D)
    # diffs:   (Q, N, D)  → sum over D → (Q, N)
    diff = queries[:, np.newaxis, :] - data[np.newaxis, :, :]  # (Q, N, D)
    sq_dists = (diff * diff).sum(axis=2)                        # (Q, N)
    return np.argsort(sq_dists, axis=1)[:, :k]                  # (Q, k)


# ── recall computation ─────────────────────────────────────────────────────────

def _mean_recall(
    gt: np.ndarray,          # (Q, k) integer row ids
    candidates: np.ndarray,  # (Q, k) integer row ids
    k: int,
) -> float:
    """Mean recall@k over Q queries."""
    recalls = []
    for q in range(gt.shape[0]):
        gt_set = set(gt[q].tolist())
        cand_set = set(candidates[q].tolist())
        recalls.append(len(gt_set & cand_set) / k)
    return float(np.mean(recalls))


# ── LanceDB helpers ────────────────────────────────────────────────────────────

def _table_name(m: int) -> str:
    """Return a run-safe table name for a given num_sub_vectors value."""
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"ivf_pq_m{m}_{run_id}"


def _build_table(
    db: lancedb.DBConnection,
    name: str,
    data: np.ndarray,
) -> lancedb.table.Table:
    """Create (or overwrite) a LanceDB table from the numpy dataset."""
    ids = np.arange(len(data), dtype=np.int64)

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), data.shape[1])),
    ])

    arrow_table = pa.table(
        {
            "id": ids,
            "vector": pa.array(data.tolist(), type=pa.list_(pa.float32(), data.shape[1])),
        },
        schema=schema,
    )

    # Drop existing table if present so we start fresh.
    if name in db.table_names():
        db.drop_table(name)

    return db.create_table(name, arrow_table)


def _search_table(
    table: lancedb.table.Table,
    queries: np.ndarray,
    k: int,
    nprobes: int,
) -> np.ndarray:
    """Run ANN search for all queries; return (Q, k) array of row `id` values."""
    result_ids = []
    for qvec in queries:
        rows = (
            table.search(qvec)
            .limit(k)
            .nprobes(nprobes)
            .to_list()
        )
        result_ids.append([r["id"] for r in rows])
    return np.array(result_ids, dtype=np.int64)


# ── public API ─────────────────────────────────────────────────────────────────

def sweep() -> Dict[int, float]:
    """Run the IVF_PQ recall sweep.

    Returns
    -------
    dict[int, float]
        Mapping from num_sub_vectors → mean recall@10, for m in {4, 8, 16}.
    """
    # Load fixtures.
    data = np.load(_DATA_PATH).astype(np.float32)     # (1024, 64)
    queries = np.load(_QUERIES_PATH).astype(np.float32)  # (30, 64)

    # Compute brute-force ground truth once.
    gt = _brute_force_top_k(data, queries, _TOP_K)    # (30, 10)

    db = lancedb.connect(_DB_PATH)

    results: Dict[int, float] = {}

    for m in _M_VALUES:
        name = _table_name(m)

        # Build a fresh table for this m value.
        table = _build_table(db, name, data)

        # Create IVF_PQ index with the current num_sub_vectors.
        index_name = f"vector_ivfpq_m{m}"
        table.create_index(
            metric="l2",
            num_partitions=_NUM_PARTITIONS,
            num_sub_vectors=m,
            index_type="IVF_PQ",
            replace=True,
            name=index_name,
        )

        # Wait for the index to be fully built before querying.
        table.wait_for_index([index_name], timeout=_INDEX_TIMEOUT)

        # Run ANN queries.
        candidates = _search_table(table, queries, _TOP_K, _NPROBES)  # (30, 10)

        # Compute mean recall@10.
        recall = _mean_recall(gt, candidates, _TOP_K)
        results[m] = recall

    return results
