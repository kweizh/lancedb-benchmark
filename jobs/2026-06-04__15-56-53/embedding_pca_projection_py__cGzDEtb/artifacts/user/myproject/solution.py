"""
solution.py – PCA-backed vector search against the articles_pca_<ZEALT_RUN_ID> LanceDB table.

Public API
----------
search(query_vec, k) -> list[dict]
    Project a 128-d query vector into 16-d PCA space and return the k most
    similar rows from the compressed table, each as:
        {"id": int, "title": str, "original_id": int}
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence, Union

import numpy as np
import lancedb

# ── Constants ─────────────────────────────────────────────────────────────────
_DB_PATH    = "/home/user/myproject/lancedb"
_MODEL_PATH = "/app/pca_model.npz"
_N_COMPONENTS = 16


# ── Lazy-cached helpers ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_pca_model() -> tuple[np.ndarray, np.ndarray]:
    """Load and cache the PCA components and mean from disk."""
    data = np.load(_MODEL_PATH)
    components: np.ndarray = data["components"].astype(np.float64)  # (16, 128)
    mean: np.ndarray       = data["mean"].astype(np.float64)         # (128,)
    assert components.shape == (_N_COMPONENTS, 128), (
        f"Unexpected components shape: {components.shape}"
    )
    assert mean.shape == (128,), f"Unexpected mean shape: {mean.shape}"
    return components, mean


@lru_cache(maxsize=1)
def _get_table():
    """Open and cache the PCA LanceDB table."""
    run_id    = os.environ["ZEALT_RUN_ID"]
    table_name = f"articles_pca_{run_id}"
    db  = lancedb.connect(_DB_PATH)
    tbl = db.open_table(table_name)
    return tbl


# ── PCA projection helper ─────────────────────────────────────────────────────

def _project(query_vec: Union[Sequence[float], np.ndarray]) -> list[float]:
    """Project a 128-d vector into 16-d PCA space.

    Follows the sklearn convention:
        projected = (query - mean) @ components.T
    which is equivalent to  components @ (query - mean).
    """
    q = np.asarray(query_vec, dtype=np.float64)
    if q.shape != (128,):
        raise ValueError(f"query_vec must have length 128, got {q.shape}")
    components, mean = _load_pca_model()
    projected = (q - mean) @ components.T          # (16,)
    return projected.astype(np.float32).tolist()


# ── Public API ────────────────────────────────────────────────────────────────

def search(
    query_vec: Union[Sequence[float], np.ndarray],
    k: int,
) -> list[dict]:
    """Search the 16-d PCA table for the k nearest neighbours of query_vec.

    Parameters
    ----------
    query_vec : array-like of length 128
        The query embedding in the original 128-d space.
    k : int
        Number of results to return.

    Returns
    -------
    list of dict
        Each dict has keys ``id`` (int), ``title`` (str), ``original_id`` (int),
        ordered from most to least similar.
    """
    projected = _project(query_vec)

    tbl     = _get_table()
    results = (
        tbl.search(projected)
           .limit(k)
           .select(["id", "title", "original_id"])
           .metric("l2")
           .to_pandas()
    )

    return [
        {
            "id":          int(row["id"]),
            "title":       str(row["title"]),
            "original_id": int(row["original_id"]),
        }
        for _, row in results.iterrows()
    ]
