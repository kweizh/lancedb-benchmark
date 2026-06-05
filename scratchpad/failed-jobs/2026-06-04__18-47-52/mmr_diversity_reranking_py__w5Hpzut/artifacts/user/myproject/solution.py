import os
import numpy as np
import pyarrow as pa
import lancedb

def get_table_name():
    run_id = os.environ["ZEALT_RUN_ID"]
    return f"mmr_docs_{run_id}"

def get_db_path():
    return "/app/db"

def build_dataset() -> None:
    rng = np.random.default_rng(seed=2026)
    A = rng.standard_normal((32, 32))
    Q, _ = np.linalg.qr(A)
    
    data = []
    for c in range(10):
        centroid = Q[:, c]
        for j in range(12):
            noise = rng.standard_normal(32)
            row_vec = centroid + 0.05 * noise
            row_vec = row_vec.astype(np.float32)
            
            doc_id = c * 12 + j
            
            data.append({
                "id": doc_id,
                "cluster_id": c,
                "vector": row_vec
            })
            
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("cluster_id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 32))
    ])
    
    db = lancedb.connect(get_db_path())
    table_name = get_table_name()
    
    db.create_table(table_name, data=data, schema=schema, mode="overwrite")

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def mmr_search(query_vec, k=10, lambda_=0.5) -> list[int]:
    db = lancedb.connect(get_db_path())
    table_name = get_table_name()
    table = db.open_table(table_name)
    
    # Candidate pool: top-30 rows
    results = table.search(query_vec).distance_type("cosine").limit(30).to_list()
    
    candidates = []
    for r in results:
        candidates.append({
            "id": r["id"],
            "vector": np.array(r["vector"], dtype=np.float64)
        })
        
    selected_ids = []
    selected_vectors = []
    
    query_vec_np = np.asarray(query_vec, dtype=np.float64)
    
    while len(selected_ids) < k and candidates:
        best_score = -float('inf')
        best_idx = -1
        
        for i, cand in enumerate(candidates):
            sim_q_d = cosine_similarity(query_vec_np, cand["vector"])
            
            if not selected_vectors:
                redundancy = 0.0
            else:
                redundancy = max(cosine_similarity(cand["vector"], sv) for sv in selected_vectors)
                
            score = lambda_ * sim_q_d - (1.0 - lambda_) * redundancy
            
            if score > best_score:
                best_score = score
                best_idx = i
                
        best_cand = candidates.pop(best_idx)
        selected_ids.append(best_cand["id"])
        selected_vectors.append(best_cand["vector"])
        
    return selected_ids

if __name__ == "__main__":
    build_dataset()
