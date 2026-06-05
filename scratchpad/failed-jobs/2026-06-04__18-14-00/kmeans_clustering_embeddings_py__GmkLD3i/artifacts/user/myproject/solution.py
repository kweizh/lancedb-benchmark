"""
solution.py – Public API for the K-means clustering results stored in LanceDB.

Exported callables
------------------
cluster_centroids() -> numpy.ndarray
    Returns an (8, 32) float32 array of centroids in cluster-id order.

nearest_cluster(query_vec) -> int
    Returns the cluster_id (0..7) of the centroid nearest to query_vec,
    computed via a LanceDB cosine vector search on the `centroids` table.
"""

import numpy as np
import lancedb

LANCEDB_PATH = "/home/user/myproject/lancedb_data"
VECTOR_DIM = 32
N_CLUSTERS = 8

# ------------------------------------------------------------------ #
# Module-level lazy connection (re-used across calls)
# ------------------------------------------------------------------ #
_db = None
_centroids_table = None


def _get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(LANCEDB_PATH)
    return _db


def _get_centroids_table():
    global _centroids_table
    if _centroids_table is None:
        _centroids_table = _get_db().open_table("centroids")
    return _centroids_table


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def cluster_centroids() -> np.ndarray:
    """
    Return the 8 cluster centroids as a numpy array of shape (8, 32),
    dtype float32, sorted by cluster_id.
    """
    table = _get_centroids_table()
    df = table.to_pandas().sort_values("cluster_id").reset_index(drop=True)
    centroids = np.vstack(df["vector"].values).astype(np.float32)
    return centroids  # shape (8, 32)


def nearest_cluster(query_vec) -> int:
    """
    Return the cluster_id (int, 0..7) of the centroid nearest to query_vec.

    Parameters
    ----------
    query_vec : array-like of length 32
        The query vector (Python list or numpy array).
    """
    query = np.asarray(query_vec, dtype=np.float32).flatten().tolist()
    table = _get_centroids_table()
    results = (
        table.search(query, vector_column_name="vector")
             .metric("cosine")
             .limit(1)
             .to_pandas()
    )
    return int(results["cluster_id"].iloc[0])
