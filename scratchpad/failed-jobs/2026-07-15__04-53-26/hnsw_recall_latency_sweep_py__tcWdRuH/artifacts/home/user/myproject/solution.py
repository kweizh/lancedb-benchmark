"""LanceDB ANN recall / latency parameter sweep.

This module exposes ``build_index`` and ``sweep`` for evaluating how an
IVF_PQ index's recall@k and mean per-query latency change as search-time
``nprobes`` and ``refine_factor`` are varied on a caller-supplied grid.

The implementation reads the LanceDB URI and run id from environment
variables (``LANCEDB_URI`` and ``ZEALT_RUN_ID``) and operates on a
benchmark table named ``vectors_<RUN_ID>`` containing an integer ``id``
column and a fixed-size list ``vector`` column of float32 values.
"""

from __future__ import annotations

import itertools
import os
import time
from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import lancedb

# Sensible defaults for an exhaustive brute-force scan over up to a few
# thousand rows per query.
_BRUTE_FORCE_CHUNK = 4096

# Index parameters required by the task spec.
_NUM_PARTITIONS = 64
_NUM_SUB_VECTORS = 8
_METRIC = "l2"
_VECTOR_COLUMN = "vector"

# Default name lancedb assigns when none is supplied to ``create_index``.
_DEFAULT_INDEX_NAME = "vector_idx"


def _table_name() -> str:
    """Return the benchmark table name derived from ``ZEALT_RUN_ID``."""
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    if not run_id:
        raise RuntimeError("ZEALT_RUN_ID environment variable is not set")
    return f"vectors_{run_id}"


def _db() -> lancedb.LanceDBConnection:
    uri = os.environ.get("LANCEDB_URI", "/app/lancedb")
    return lancedb.connect(uri)


def _open_table():
    """Open the benchmark table, raising a clear error if it is missing."""
    return _db().open_table(_table_name())


def _index_names(table) -> List[str]:
    """Return the list of existing index names on the table."""
    return [idx.name for idx in table.list_indices()]


def build_index() -> None:
    """(Re)build the IVF_PQ ANN index on the ``vector`` column.

    Uses ``replace=True`` so it is safe to call repeatedly: any prior
    vector index on the table is dropped first. The call blocks on the
    underlying index build via ``wait_for_index`` so that searches
    invoked immediately afterwards are served by the new index.
    """
    table = _open_table()
    table.create_index(
        metric=_METRIC,
        num_partitions=_NUM_PARTITIONS,
        num_sub_vectors=_NUM_SUB_VECTORS,
        vector_column_name=_VECTOR_COLUMN,
        replace=True,
        index_type="IVF_PQ",
    )

    # create_index returns immediately while training / writing the index
    # parts continues in the background. Block until the index is queryable.
    existing = _index_names(table)
    names: List[str] = existing if existing else [_DEFAULT_INDEX_NAME]
    # ``wait_for_index`` accepts the names explicitly; passing an empty
    # iterable causes it to wait on nothing, which we never want here.
    if names:
        table.wait_for_index(names, timeout=timedelta(minutes=10))

    # Make sure subsequent reads see the freshly-built index metadata.
    table.checkout_latest()


def _load_all_vectors(table):
    """Return ``(ids, vectors)`` for the entire table as numpy arrays.

    Used to compute brute-force exact top-k ground truth deterministically
    without round-tripping every query through LanceDB.
    """
    arrow_tbl = table.to_arrow()
    ids = np.asarray(arrow_tbl.column("id").to_pylist(), dtype=np.int64)
    vectors = np.stack(
        [np.asarray(v, dtype=np.float32) for v in arrow_tbl.column("vector").to_pylist()]
    )
    return ids, vectors


def _exact_top_k(query: np.ndarray, ids: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """Return the ids of the exact top-``k`` nearest neighbours by L2.

    Computed in numpy so tie-breaking by ascending ``id`` is stable across
    runs (np.argsort with ``kind="stable"`` preserves insertion order,
    and ``ids`` are 0..N-1 in ascending insertion order).
    """
    # Compute squared L2 distances; the ordering is identical to L2.
    diff = vectors - query.astype(np.float32, copy=False)
    sq = np.einsum("ij,ij->i", diff, diff)
    # ``stable`` sort keeps items with equal distances in their original
    # array order, and ``ids`` are stored strictly ascending (0..N-1), so
    # distance ties break by ascending id deterministically.
    order = np.argsort(sq, kind="stable")[:k]
    return ids[order]


def _ann_top_k(table, query: np.ndarray, k: int, nprobes: int, refine_factor: int) -> np.ndarray:
    """Return the ids of the top-``k`` ANN search results for ``query``."""
    res = (
        table.search(query)
        .metric(_METRIC)
        .nprobes(nprobes)
        .refine_factor(refine_factor)
        .limit(k)
        .to_arrow()
    )
    return np.asarray(res.column("id").to_pylist(), dtype=np.int64)


def sweep(
    query_set: np.ndarray,
    param_grid: Dict[str, List[int]],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Run the recall / latency sweep and return one row per configuration.

    Parameters
    ----------
    query_set:
        2-D float32 array of shape ``(num_queries, dim)``.
    param_grid:
        Dict with ``"nprobes"`` and ``"refine_factor"`` keys, each a list of
        positive integers. The full Cartesian product is evaluated.
    k:
        Recall is computed at this ``k``.

    Returns
    -------
    list of dict
        One entry per (nprobes, refine_factor) configuration, sorted
        ascending by ``(nprobes, refine_factor)``. Each dict has keys
        ``"nprobes"`` (int), ``"refine_factor"`` (int), ``"recall"``
        (float in [0.0, 1.0]) and ``"mean_latency_ms"`` (float > 0).
    """
    if query_set is None or len(query_set) == 0:
        raise ValueError("query_set must be a non-empty 2-D float32 array")
    query_set = np.ascontiguousarray(query_set, dtype=np.float32)
    if query_set.ndim != 2:
        raise ValueError("query_set must be a 2-D array of shape (num_queries, dim)")

    nprobes_list = list(param_grid.get("nprobes", []))
    refine_list = list(param_grid.get("refine_factor", []))
    if not nprobes_list or not refine_list:
        raise ValueError("param_grid must contain non-empty 'nprobes' and 'refine_factor' lists")

    table = _open_table()
    # Make absolutely sure no cached / partial state from a prior run affects
    # this sweep (the task says calling sweep twice must be deterministic).
    table.checkout_latest()

    ids, vectors = _load_all_vectors(table)

    num_queries = query_set.shape[0]
    results: List[Dict[str, Any]] = []

    # Sort ascending on both keys so the returned list is in a stable order.
    configs = sorted(
        ((np_val, rf_val) for np_val, rf_val in itertools.product(nprobes_list, refine_list)),
        key=lambda cfg: (cfg[0], cfg[1]),
    )

    # Precompute exact top-k once per query so recall evaluation is cheap.
    exact_neighbors: List[np.ndarray] = [
        _exact_top_k(query_set[i], ids, vectors, k) for i in range(num_queries)
    ]

    for nprobes, refine_factor in configs:
        recalls: List[float] = []
        latencies_ms: List[float] = []

        for i in range(num_queries):
            t0 = time.perf_counter()
            ann_ids = _ann_top_k(table, query_set[i], k, nprobes, refine_factor)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Intersection size / k → per-query recall@k.
            exact_set = set(int(x) for x in exact_neighbors[i].tolist())
            hits = sum(1 for x in ann_ids.tolist() if int(x) in exact_set)
            recalls.append(hits / float(k))
            latencies_ms.append(elapsed_ms)

        results.append(
            {
                "nprobes": int(nprobes),
                "refine_factor": int(refine_factor),
                "recall": float(np.mean(recalls)),
                "mean_latency_ms": float(np.mean(latencies_ms)),
            }
        )

    return results
