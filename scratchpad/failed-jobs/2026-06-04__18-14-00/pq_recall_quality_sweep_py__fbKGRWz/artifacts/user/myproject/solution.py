"""
IVF_PQ recall-quality sweep over num_sub_vectors (m = 4, 8, 16).

sweep() → dict[int, float]  mapping  m → mean recall@10
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict

import lancedb
import numpy as np
import pyarrow as pa

# ── Fixed paths ────────────────────────────────────────────────────────────────
_DB_PATH = "/home/user/myproject/lancedb_data/"
_DATA_PATH = "/app/fixtures/data.npy"
_QUERIES_PATH = "/app/fixtures/queries.npy"

# ── Sweep configuration ────────────────────────────────────────────────────────
_M_VALUES = [4, 8, 16]
_NUM_PARTITIONS = 16
_NPROBES = 16
_TOP_K = 10


# ── Brute-force ground truth ───────────────────────────────────────────────────

def _brute_force_top_k(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Return (num_queries, k) array of row indices for the top-k nearest
    neighbours of each query under squared L2 distance."""
    # data:    (N, D)   queries: (Q, D)
    # squared L2:  ||q - d||^2 = ||q||^2 - 2*q@d.T + ||d||^2
    q_sq = (queries ** 2).sum(axis=1, keepdims=True)          # (Q, 1)
    d_sq = (data ** 2).sum(axis=1, keepdims=True).T            # (1, N)
    dists = q_sq - 2.0 * (queries @ data.T) + d_sq             # (Q, N)
    return np.argsort(dists, axis=1)[:, :k]                     # (Q, k)


# ── Recall helper ──────────────────────────────────────────────────────────────

def _recall_at_k(
    candidates: list[list[int]],
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Mean recall@k over all queries."""
    recalls = []
    for cands, gt in zip(candidates, ground_truth):
        cand_set = set(cands[:k])
        gt_set = set(gt[:k].tolist())
        recalls.append(len(cand_set & gt_set) / k)
    return float(np.mean(recalls))


# ── Main sweep ─────────────────────────────────────────────────────────────────

def sweep() -> Dict[int, float]:
    """Run IVF_PQ recall sweep for m in {4, 8, 16}.

    Returns
    -------
    dict[int, float]
        Mapping from num_sub_vectors value to mean recall@10.
    """
    run_id = os.environ.get("ZEALT_RUN_ID", "default")

    # Load dataset and queries
    data = np.load(_DATA_PATH).astype(np.float32)   # (1024, 64)
    queries = np.load(_QUERIES_PATH).astype(np.float32)  # (30, 64)

    n, dim = data.shape
    ids = np.arange(n, dtype=np.int64)

    # Brute-force ground truth (stable, computed once)
    gt_indices = _brute_force_top_k(data, queries, k=_TOP_K)  # (30, 10)

    # Connect to LanceDB
    db = lancedb.connect(_DB_PATH)

    results: Dict[int, float] = {}

    for m in _M_VALUES:
        table_name = f"ivf_pq_m{m}_{run_id}"
        index_name = f"idx_m{m}"

        # Build a PyArrow table: id (int64) + vector (fixed_size_list<float32>[dim])
        pa_table = pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "vector": pa.array(
                    data.tolist(),
                    type=pa.list_(pa.float32(), dim),
                ),
            }
        )

        # Create (or overwrite) the LanceDB table
        tbl = db.create_table(table_name, pa_table, mode="overwrite")

        # Build IVF_PQ index
        tbl.create_index(
            index_type="IVF_PQ",
            metric="l2",
            num_partitions=_NUM_PARTITIONS,
            num_sub_vectors=m,
            vector_column_name="vector",
            name=index_name,
            replace=True,
        )

        # Wait until the index is fully built
        tbl.wait_for_index([index_name], timeout=timedelta(seconds=300))

        # Query each of the 30 query vectors and collect top-10 ids
        candidate_ids: list[list[int]] = []
        for qvec in queries:
            rows = (
                tbl.search(qvec.tolist())
                .limit(_TOP_K)
                .nprobes(_NPROBES)
                .to_list()
            )
            candidate_ids.append([r["id"] for r in rows])

        recall = _recall_at_k(candidate_ids, gt_indices, k=_TOP_K)
        results[m] = recall
        print(f"  m={m:2d}  recall@{_TOP_K} = {recall:.4f}")

    return results
