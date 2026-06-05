import lancedb
import numpy as np

DB_PATH = "/home/user/myproject/lancedb_data"

# Lazily initialize the database connection and tables
_db = None
_centroids_tbl = None

def _get_centroids_table():
    global _db, _centroids_tbl
    if _db is None:
        _db = lancedb.connect(DB_PATH)
    if _centroids_tbl is None:
        _centroids_tbl = _db.open_table("centroids")
    return _centroids_tbl

def cluster_centroids() -> np.ndarray:
    """
    Returns an array of shape (8, 32) and dtype float32 containing the centroids
    in cluster-id order (row i = cluster i).
    """
    tbl = _get_centroids_table()
    df = tbl.to_pandas().sort_values("cluster_id")
    centroids = np.vstack(df["vector"].values).astype(np.float32)
    return centroids

def nearest_cluster(query_vec) -> int:
    """
    Accepts a length-32 1-D vector (Python list or numpy array) and returns
    the cluster id (0..7) of the nearest centroid, computed via a LanceDB
    cosine vector search against the centroids table.
    """
    tbl = _get_centroids_table()
    
    # Ensure query_vec is passed as a 1D list/array
    if isinstance(query_vec, np.ndarray):
        query_vec = query_vec.flatten().astype(np.float32).tolist()
    elif isinstance(query_vec, list):
        # Ensure elements are float
        query_vec = [float(x) for x in query_vec]
        
    res = tbl.search(query_vec).metric("cosine").limit(1).to_pandas()
    if res.empty:
        raise ValueError("Centroids table is empty or search failed.")
    
    return int(res["cluster_id"].iloc[0])
