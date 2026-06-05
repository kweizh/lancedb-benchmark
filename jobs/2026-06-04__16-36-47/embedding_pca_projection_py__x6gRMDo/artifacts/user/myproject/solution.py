import os
import lancedb
import numpy as np

# Cache the model to avoid reloading on every search
_pca_model = None

def _get_pca_model():
    global _pca_model
    if _pca_model is None:
        data = np.load("/app/pca_model.npz")
        _pca_model = {
            "components": data["components"],
            "mean": data["mean"]
        }
    return _pca_model

def search(query_vec, k):
    model = _get_pca_model()
    
    # Ensure query_vec is a numpy array
    query_vec = np.asarray(query_vec, dtype=np.float32)
    
    # Project: (x - mean) @ components.T
    projected_vec = np.dot(query_vec - model["mean"], model["components"].T).astype(np.float32)
    
    # Connect to LanceDB
    db_path = "/home/user/myproject/lancedb"
    db = lancedb.connect(db_path)
    
    run_id = os.environ.get("ZEALT_RUN_ID")
    table_name = f"articles_pca_{run_id}"
    
    table = db.open_table(table_name)
    
    # Perform search
    results = table.search(projected_vec).limit(k).to_list()
    
    # Return list of dicts with id, title, original_id
    out = []
    for r in results:
        out.append({
            "id": int(r["id"]),
            "title": r["title"],
            "original_id": int(r["original_id"])
        })
        
    return out
