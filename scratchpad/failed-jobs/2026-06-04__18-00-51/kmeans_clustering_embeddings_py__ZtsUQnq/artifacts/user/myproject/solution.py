"""
solution.py – public API for the K-means clustering results.

Exposes:
    cluster_centroids() -> numpy.ndarray  shape (8, 32) float32
    nearest_cluster(query_vec)            -> int  (0..7)
"""

import numpy as np
import lancedb

LANCEDB_DIR = "/home/user/myproject/lancedb_data"
_VECTOR_DIM = 32
_N_CLUSTERS = 8


def cluster_centroids() -> np.ndarray:
    """Return the 8 centroids in cluster-id order as a (8, 32) float32 array."""
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("centroids")
    df = tbl.to_pandas()
    # Sort by cluster_id so row i == centroid for cluster i
    df = df.sort_values("cluster_id").reset_index(drop=True)
    vectors = np.vstack(df["vector"].values).astype("float32")  # (8, 32)
    return vectors


def nearest_cluster(query_vec) -> int:
    """
    Return the cluster_id (0..7) of the nearest centroid to `query_vec`,
    using an L2 (Euclidean) vector search on the `centroids` LanceDB table.
    L2 matches the distance metric used by sklearn KMeans.predict.

    Parameters
    ----------
    query_vec : array-like, length 32
    """
    query = np.asarray(query_vec, dtype="float32").flatten().tolist()
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("centroids")
    results = (
        tbl.search(query, vector_column_name="vector")
           .metric("l2")
           .limit(1)
           .to_pandas()
    )
    return int(results["cluster_id"].iloc[0])
