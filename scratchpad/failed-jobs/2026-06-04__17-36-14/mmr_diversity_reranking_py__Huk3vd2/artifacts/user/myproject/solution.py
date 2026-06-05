import os
import numpy as np
import pyarrow as pa
import lancedb

def build_dataset() -> None:
    """
    Builds the deterministic fixture table described in the specification at /app/db.
    The table name is mmr_docs_${ZEALT_RUN_ID}.
    """
    db_dir = "/app/db"
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"mmr_docs_{run_id}"
    
    # Deterministic generation using numpy.random.default_rng(seed=2026)
    rng = np.random.default_rng(seed=2026)
    
    # 1. Draw a (32, 32) standard-normal matrix A and compute Q, _ = np.linalg.qr(A)
    A = rng.standard_normal((32, 32))
    Q, _ = np.linalg.qr(A)
    
    data = []
    # 2. For each cluster c in range(10) and slot j in range(12)...
    for c in range(10):
        centroid = Q[:, c]
        for j in range(12):
            # Draw one fresh (32,) standard-normal noise vector from the same RNG
            noise = rng.standard_normal(32)
            row_vec = centroid + 0.05 * noise
            row_vec_f32 = row_vec.astype(np.float32)
            
            # id = c * 12 + j and cluster_id = c
            row_id = c * 12 + j
            data.append({
                "id": row_id,
                "cluster_id": c,
                "vector": row_vec_f32.tolist()
            })
            
    # Connect to LanceDB and create table with Arrow schema
    db = lancedb.connect(db_dir)
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("cluster_id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 32))
    ])
    
    db.create_table(table_name, data=data, schema=schema, mode="overwrite")
    print(f"Dataset successfully built with 120 rows in table '{table_name}'.")

def mmr_search(query_vec, k=10, lambda_=0.5) -> list[int]:
    """
    Runs MMR re-ranking against the table mmr_docs_${ZEALT_RUN_ID} and returns
    the selected document id values as a Python list[int] of length k,
    in the order they were picked by MMR.
    """
    db_dir = "/app/db"
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"mmr_docs_{run_id}"
    
    db = lancedb.connect(db_dir)
    table = db.open_table(table_name)
    
    # Candidate pool: top-30 rows from a single cosine vector search against the query
    candidates = table.search(query_vec).distance_type("cosine").limit(30).to_list()
    if not candidates:
        return []
        
    # Convert query to unit vector (using float64 for precision)
    q_arr = np.array(query_vec, dtype=np.float64)
    q_norm = np.linalg.norm(q_arr)
    if q_norm > 1e-12:
        q_unit = q_arr / q_norm
    else:
        q_unit = q_arr
        
    # Convert candidates to unit vectors (using float64 for precision)
    cand_units = []
    cand_ids = []
    for c in candidates:
        cand_ids.append(c["id"])
        v = np.array(c["vector"], dtype=np.float64)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            cand_units.append(v / v_norm)
        else:
            cand_units.append(v)
            
    n_candidates = len(candidates)
    
    # Compute relevance term for all candidates: sim(q, d) = dot(q_unit, d_unit)
    relevances = [np.dot(q_unit, u) for u in cand_units]
    
    # Compute similarity matrix between all candidates: sim(d, d') = dot(d_unit, d_prime_unit)
    similarities = np.zeros((n_candidates, n_candidates), dtype=np.float64)
    for i in range(n_candidates):
        for j in range(n_candidates):
            similarities[i, j] = np.dot(cand_units[i], cand_units[j])
            
    selected_indices = []
    for _ in range(min(k, n_candidates)):
        best_score = -float("inf")
        best_idx = -1
        for idx in range(n_candidates):
            if idx in selected_indices:
                continue
                
            relevance = relevances[idx]
            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(similarities[idx, sel_idx] for sel_idx in selected_indices)
                
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
                
        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        
    return [cand_ids[idx] for idx in selected_indices]

if __name__ == "__main__":
    build_dataset()
