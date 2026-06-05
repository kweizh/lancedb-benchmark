"""
LanceDB MMR Diversity Re-ranking
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


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset() -> None:
    """Build the deterministic 120-row fixture table and write it to /app/db."""

    rng = np.random.default_rng(seed=2026)

    # Step 1 – derive 10 orthonormal centroids from a random QR decomposition.
    A = rng.standard_normal((32, 32))
    Q, _ = np.linalg.qr(A)          # Q is (32, 32); Q[:, c] is centroid c

    # Step 2 – generate rows in cluster-major, j-major order.
    ids: list[int] = []
    cluster_ids: list[int] = []
    vectors: list[list[float]] = []

    for c in range(10):
        centroid = Q[:, c]           # shape (32,)
        for j in range(12):
            noise = rng.standard_normal(32)
            row_vec = (centroid + 0.05 * noise).astype(np.float32)
            ids.append(c * 12 + j)
            cluster_ids.append(c)
            vectors.append(row_vec.tolist())

    # Step 3 – build Arrow table with the required schema.
    schema = pa.schema([
        pa.field("id",         pa.int64()),
        pa.field("cluster_id", pa.int64()),
        pa.field("vector",     pa.list_(pa.float32(), 32)),
    ])

    table_data = pa.table(
        {
            "id":         pa.array(ids,         type=pa.int64()),
            "cluster_id": pa.array(cluster_ids, type=pa.int64()),
            "vector":     pa.array(vectors,     type=pa.list_(pa.float32(), 32)),
        },
        schema=schema,
    )

    # Step 4 – write to LanceDB (overwrite so re-runs are safe).
    db = _connect()
    db.create_table(_table_name(), data=table_data, mode="overwrite")
    print(f"[build_dataset] table '{_table_name()}' written with {len(ids)} rows.")


# ---------------------------------------------------------------------------
# MMR search
# ---------------------------------------------------------------------------

def mmr_search(
    query_vec,
    k: int = 10,
    lambda_: float = 0.5,
) -> list[int]:
    """
    Run MMR re-ranking against the LanceDB table and return k selected ids.

    Parameters
    ----------
    query_vec : array-like of length 32
    k         : number of results to return (default 10)
    lambda_   : trade-off between relevance (1.0) and diversity (0.0)

    Returns
    -------
    list[int] of length k, in MMR selection order (most relevant first).
    """
    q = np.asarray(query_vec, dtype=np.float64)

    # -----------------------------------------------------------------------
    # 1. Retrieve candidate pool: top-30 cosine nearest neighbours.
    # -----------------------------------------------------------------------
    db = _connect()
    tbl = db.open_table(_table_name())

    rows = (
        tbl.search(q.astype(np.float32).tolist())
        .distance_type("cosine")
        .limit(30)
        .to_list()
    )

    # -----------------------------------------------------------------------
    # 2. Extract stored vectors and ids from the result dicts.
    # -----------------------------------------------------------------------
    cand_ids: list[int] = []
    cand_vecs: list[np.ndarray] = []

    for row in rows:
        cand_ids.append(int(row["id"]))
        vec = np.asarray(row["vector"], dtype=np.float64)
        cand_vecs.append(vec)

    n = len(cand_ids)

    # -----------------------------------------------------------------------
    # 3. Pre-compute normalised vectors (once) for fast dot-product similarity.
    # -----------------------------------------------------------------------
    def _unit(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0.0 else v

    q_unit = _unit(q)
    cand_units = [_unit(v) for v in cand_vecs]

    # cosine similarity between query and every candidate
    rel_scores = np.array([float(np.dot(q_unit, u)) for u in cand_units])

    # -----------------------------------------------------------------------
    # 4. MMR iterative selection.
    # -----------------------------------------------------------------------
    selected_indices: list[int] = []   # indices into cand_ids / cand_units
    remaining: set[int] = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_score = -np.inf

        for i in remaining:
            relevance = rel_scores[i]

            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(
                    float(np.dot(cand_units[i], cand_units[s]))
                    for s in selected_indices
                )

            score = lambda_ * relevance - (1.0 - lambda_) * redundancy

            if score > best_score:
                best_score = score
                best_idx = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [cand_ids[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_dataset()
