import os
import lancedb
import numpy as np

# Global cache
_PCA_MODEL = None
_DB_TABLE = None

def _get_pca_model():
    global _PCA_MODEL
    if _PCA_MODEL is None:
        model_path = "/app/pca_model.npz"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PCA model file not found at {model_path}")
        data = np.load(model_path)
        _PCA_MODEL = {
            "components": data["components"],
            "mean": data["mean"]
        }
    return _PCA_MODEL

def _get_table():
    global _DB_TABLE
    if _DB_TABLE is None:
        db_path = "/home/user/myproject/lancedb/"
        db = lancedb.connect(db_path)
        zealt_run_id = os.environ.get("ZEALT_RUN_ID")
        if not zealt_run_id:
            raise ValueError("ZEALT_RUN_ID environment variable is not set!")
        table_name = f"articles_pca_{zealt_run_id}"
        _DB_TABLE = db.open_table(table_name)
    return _DB_TABLE

def search(query_vec, k):
    """
    Projects query_vec (128-d) to 16-d using the persisted PCA model,
    runs a vector similarity search against the 16-d table,
    and returns a JSON-serializable Python list of length k.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
        
    # Convert query_vec to numpy array
    query_arr = np.array(query_vec, dtype=np.float32)
    if query_arr.shape != (128,):
        raise ValueError(f"query_vec must be 128-dimensional, got shape {query_arr.shape}")
        
    # Load PCA model and project
    pca_model = _get_pca_model()
    components = pca_model["components"]
    mean = pca_model["mean"]
    
    # Project: (query_vec - mean) @ components.T
    projected_vec = (query_arr - mean) @ components.T
    projected_list = projected_vec.tolist()
    
    # Get table and search
    tbl = _get_table()
    results = tbl.search(projected_list).limit(k).to_list()
    
    # Format to JSON-serializable list of dicts with keys: id, title, original_id
    output = []
    for row in results:
        output.append({
            "id": int(row["id"]),
            "title": str(row["title"]),
            "original_id": int(row["original_id"])
        })
    return output
