import lancedb
import numpy as np

DB_PATH = "/home/user/myproject/lancedb_data"

def cluster_centroids() -> np.ndarray:
    db = lancedb.connect(DB_PATH)
    centroids_table = db.open_table("centroids")
    df = centroids_table.to_pandas()
    # Sort by cluster_id just in case
    df = df.sort_values("cluster_id")
    return np.vstack(df["vector"].values).astype(np.float32)

def nearest_cluster(query_vec) -> int:
    db = lancedb.connect(DB_PATH)
    centroids_table = db.open_table("centroids")
    query_vec = np.array(query_vec, dtype=np.float32)
    # Cosine vector search
    result = centroids_table.search(query_vec).metric("cosine").limit(1).to_pandas()
    return int(result.iloc[0]["cluster_id"])
