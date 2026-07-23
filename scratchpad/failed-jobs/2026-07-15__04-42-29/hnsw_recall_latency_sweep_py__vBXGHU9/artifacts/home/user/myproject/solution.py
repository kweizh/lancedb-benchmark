"""
LanceDB ANN Recall / Latency Parameter Sweep
=============================================
Exposes two callables:
  - build_index(): (re)builds IVF_PQ index on the benchmark table's vector column.
  - sweep(query_set, param_grid, k=10): runs the Cartesian product of (nprobes, refine_factor)
    and returns recall@k + mean query latency for every configuration.
"""

import datetime
import itertools
import os
import time
from typing import Any

import lancedb
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_table() -> Any:
    """Open and return the benchmark LanceDB table."""
    uri = os.environ.get("LANCEDB_URI", "/app/lancedb")
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    table_name = f"vectors_{run_id}"

    db = lancedb.connect(uri)
    return db.open_table(table_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index() -> None:
    """
    (Re)build an IVF_PQ vector index on the benchmark table's `vector` column.

    Parameters
    ----------
    num_partitions : 64  (fixed per spec)
    num_sub_vectors : 8  (fixed per spec)
    metric           : l2 (fixed per spec)

    Safe to call more than once — replace=True drops and rebuilds the index.
    """
    tbl = _get_table()

    tbl.create_index(
        metric="l2",
        num_partitions=64,
        num_sub_vectors=8,
        vector_column_name="vector",
        replace=True,
        index_type="IVF_PQ",
    )

    # Block until the index is fully persisted and queryable.
    tbl.wait_for_index(
        ["vector_idx"],
        timeout=datetime.timedelta(seconds=600),
    )


def sweep(
    query_set: np.ndarray,
    param_grid: dict,
    k: int = 10,
) -> list[dict]:
    """
    Run a parameter sweep over the Cartesian product of (nprobes, refine_factor).

    Parameters
    ----------
    query_set  : np.ndarray of shape (num_queries, dim), dtype float32.
    param_grid : dict with keys "nprobes" and "refine_factor", each an ascending
                 list of positive integers.
    k          : number of nearest neighbours to retrieve.

    Returns
    -------
    list of dict, sorted ascending by (nprobes, refine_factor). Each dict has:
      - nprobes        (int)
      - refine_factor  (int)
      - recall         (float in [0.0, 1.0])
      - mean_latency_ms (float > 0)
    """
    tbl = _get_table()
    query_set = np.asarray(query_set, dtype=np.float32)
    num_queries = len(query_set)

    # ------------------------------------------------------------------
    # 1. Compute exact (brute-force) ground-truth top-k for every query.
    #    bypass_vector_index() forces an exhaustive scan, bypassing ANN.
    #    Tie-breaking: sort by (_distance ASC, id ASC).
    # ------------------------------------------------------------------
    exact_ids: list[set] = []
    for q in query_set:
        result = (
            tbl.search(q.tolist())
            .bypass_vector_index()
            .limit(k)
            .select(["id"])
            .to_pandas()
        )
        # Sort deterministically: primary key = distance (asc), secondary = id (asc)
        result = result.sort_values(["_distance", "id"]).head(k)
        exact_ids.append(set(result["id"].tolist()))

    # ------------------------------------------------------------------
    # 2. Enumerate the Cartesian product of search-time parameters.
    # ------------------------------------------------------------------
    nprobes_list = sorted(param_grid["nprobes"])
    refine_list = sorted(param_grid["refine_factor"])
    configurations = sorted(itertools.product(nprobes_list, refine_list))

    results = []
    for nprobes, refine_factor in configurations:
        latencies_ms: list[float] = []
        recalls: list[float] = []

        for q_idx, q in enumerate(query_set):
            t0 = time.perf_counter()
            ann_result = (
                tbl.search(q.tolist())
                .nprobes(nprobes)
                .refine_factor(refine_factor)
                .limit(k)
                .select(["id"])
                .to_pandas()
            )
            t1 = time.perf_counter()

            # Deterministic tie-breaking
            ann_result = ann_result.sort_values(["_distance", "id"]).head(k)
            ann_ids = set(ann_result["id"].tolist())

            latencies_ms.append((t1 - t0) * 1_000.0)
            recalls.append(len(ann_ids & exact_ids[q_idx]) / k)

        results.append(
            {
                "nprobes": int(nprobes),
                "refine_factor": int(refine_factor),
                "recall": float(np.mean(recalls)),
                "mean_latency_ms": float(np.mean(latencies_ms)),
            }
        )

    # Already sorted by (nprobes, refine_factor) from configurations ordering.
    return results
