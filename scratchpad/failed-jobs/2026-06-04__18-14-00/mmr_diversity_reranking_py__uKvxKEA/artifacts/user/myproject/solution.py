"""
LanceDB MMR Diversity Re-ranking
=================================
Implements build_dataset() and mmr_search() as specified.
"""

import os
import numpy as np
import pyarrow as pa
import lancedb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_name() -> str:
    run_id = os.environ["ZEALT_RUN_ID"]
    return f"mmr_docs_{run_id}"


def _connect() -> lancedb.DBConnection:
    return lancedb.connect("/app/db")


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between every row of *a* and every row of *b*.

    Parameters
    ----------
    a : (m, d) float64
    b : (n, d) float64

    Returns
    -------
    (m, n) float64  — element [i, j] = cosine_sim(a[i], b[j])
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------

def build_dataset() -> None:
    """
    Build the deterministic 120-row fixture table in /app/db.

    Schema
    ------
    id         : int64
    cluster_id : int64
    vector     : fixed_size_list<float32>[32]
    """
    rng = np.random.default_rng(seed=2026)

    # Step 1 – orthonormal centroids via QR decomposition
    A = rng.standard_normal((32, 32))
    Q, _ = np.linalg.qr(A)          # Q is (32, 32); columns are unit vectors

    # Step 2 – generate rows: cluster-major, then slot-major
    ids, cluster_ids, vectors = [], [], []
    for c in range(10):
        centroid = Q[:, c]           # shape (32,)
        for j in range(12):
            noise = rng.standard_normal(32)
            row_vec = (centroid + 0.05 * noise).astype(np.float32)
            ids.append(c * 12 + j)
            cluster_ids.append(c)
            vectors.append(row_vec)

    # Step 3 – build Arrow table
    schema = pa.schema([
        pa.field("id",         pa.int64()),
        pa.field("cluster_id", pa.int64()),
        pa.field("vector",     pa.list_(pa.float32(), 32)),
    ])

    table_pa = pa.table(
        {
            "id":         pa.array(ids,         type=pa.int64()),
            "cluster_id": pa.array(cluster_ids, type=pa.int64()),
            "vector":     pa.array(vectors,     type=pa.list_(pa.float32(), 32)),
        },
        schema=schema,
    )

    # Step 4 – write to LanceDB (overwrite if exists)
    db = _connect()
    db.create_table(_table_name(), data=table_pa, mode="overwrite")
    print(f"[build_dataset] table '{_table_name()}' created with {len(ids)} rows.")


# ---------------------------------------------------------------------------
# mmr_search
# ---------------------------------------------------------------------------

def mmr_search(query_vec, k: int = 10, lambda_: float = 0.5) -> list[int]:
    """
    Maximal Marginal Relevance re-ranking over a LanceDB cosine vector index.

    Parameters
    ----------
    query_vec : array-like of length 32
    k         : number of results to return
    lambda_   : trade-off weight (1.0 = pure relevance, 0.0 = pure diversity)

    Returns
    -------
    list[int] of length k — document ids in MMR selection order
    """
    query_vec = np.asarray(query_vec, dtype=np.float64)

    # --- 1. Fetch candidate pool (top-30 by cosine distance) ----------------
    db    = _connect()
    tbl   = db.open_table(_table_name())
    rows  = (
        tbl.search(query_vec.astype(np.float32))
           .distance_type("cosine")
           .limit(30)
           .to_list()
    )

    # --- 2. Materialise candidate data as numpy arrays ----------------------
    cand_ids     = [r["id"]         for r in rows]           # list[int]
    cand_vecs    = np.array([r["vector"] for r in rows],
                            dtype=np.float64)                # (30, 32)

    n_cand = len(cand_ids)
    k      = min(k, n_cand)

    # --- 3. Pre-compute query–candidate cosine similarities -----------------
    # shape (n_cand,)
    q_norm   = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    c_norms  = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12
    cand_norm = cand_vecs / c_norms                          # (30, 32)

    rel_scores = cand_norm @ q_norm                          # (30,)

    # Pre-compute all pairwise cosine similarities among candidates
    # sim_matrix[i, j] = cosine_sim(cand[i], cand[j])
    sim_matrix = cand_norm @ cand_norm.T                     # (30, 30)

    # --- 4. Greedy MMR loop -------------------------------------------------
    selected_indices  = []         # indices into cand_ids / cand_vecs
    remaining_indices = list(range(n_cand))

    while len(selected_indices) < k:
        if not selected_indices:
            # First pick: maximise relevance (no redundancy term yet)
            mmr_scores = lambda_ * rel_scores[remaining_indices]
        else:
            # redundancy[i] = max_{j in selected} sim(cand[i], cand[j])
            sel_arr    = np.array(selected_indices)          # (|S|,)
            rem_arr    = np.array(remaining_indices)         # (|R|,)
            # sub-matrix: rows = remaining, cols = selected
            redundancy = sim_matrix[np.ix_(rem_arr, sel_arr)].max(axis=1)
            mmr_scores = (
                lambda_        * rel_scores[rem_arr]
                - (1 - lambda_) * redundancy
            )

        best_local = int(np.argmax(mmr_scores))
        best_idx   = remaining_indices[best_local]

        selected_indices.append(best_idx)
        remaining_indices.pop(best_local)

    return [cand_ids[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_dataset()
