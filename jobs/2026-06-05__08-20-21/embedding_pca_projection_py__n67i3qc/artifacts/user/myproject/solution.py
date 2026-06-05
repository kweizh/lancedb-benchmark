import lancedb
import numpy as np
import os
import json

class PCASearcher:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PCASearcher, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = "/home/user/myproject/lancedb/"
        self.pca_model_path = "/app/pca_model.npz"
        run_id = os.environ.get("ZEALT_RUN_ID")
        self.table_name = f"articles_pca_{run_id}"
        
        # Load PCA model
        model = np.load(self.pca_model_path)
        self.components = model['components'] # (16, 128)
        self.mean = model['mean'] # (128,)
        
        # Connect to DB
        self.db = lancedb.connect(self.db_path)
        self.table = self.db.open_table(self.table_name)
        
        self._initialized = True

    def project(self, query_vec):
        query_vec = np.array(query_vec)
        # sklearn PCA transformation: (X - mean) @ components.T
        projected = (query_vec - self.mean) @ self.components.T
        return projected.astype(np.float32)

    def search(self, query_vec, k):
        projected_vec = self.project(query_vec)
        results = self.table.search(projected_vec).limit(k).to_list()
        
        # Format results as requested
        formatted_results = []
        for res in results:
            formatted_results.append({
                "id": int(res["id"]),
                "title": str(res["title"]),
                "original_id": int(res["original_id"])
            })
        return formatted_results

def search(query_vec, k):
    searcher = PCASearcher()
    return searcher.search(query_vec, k)
