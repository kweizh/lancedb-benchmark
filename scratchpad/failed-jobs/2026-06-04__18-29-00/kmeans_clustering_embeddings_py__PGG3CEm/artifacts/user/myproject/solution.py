"""Solution module exposing cluster_centroids() and nearest_cluster() APIs."""

import numpy as np
import lancedb

DB_PATH = "/home/user/myproject/lancedb_data"


def cluster_centroids() -> np.ndarray:
    """Return an (8, 32) float32 array of centroids in cluster-id order."""
    db = lancedb.connect(DB_PATH)
    centroids_tbl = db.open_table("centroids")
    df = centroids_tbl.to_pandas()
    # Sort by cluster_id to guarantee order
    df = df.sort_values("cluster_id").reset_index(drop=True)
    centroids = np.vstack(df["vector"].values).astype(np.float32)
    return centroids


def nearest_cluster(query_vec) -> int:
    """Return the cluster id (0..7) of the nearest centroid via LanceDB cosine search."""
    db = lancedb.connect(DB_PATH)
    centroids_tbl = db.open_table("centroids")

    # Ensure query is a list of floats for LanceDB search
    query_list = np.asarray(query_vec, dtype=np.float32).flatten().tolist()

    results = centroids_tbl.search(query_list).metric("cosine").limit(1).to_pandas()
    return int(results["cluster_id"].iloc[0])