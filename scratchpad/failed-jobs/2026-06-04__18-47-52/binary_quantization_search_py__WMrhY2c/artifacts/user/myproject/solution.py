import numpy as np
import lancedb
import pyarrow as pa
import os
import datetime

def generate_data(num_rows=1024, dim=384, num_clusters=10, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    
    cluster_assignments = rng.integers(0, num_clusters, size=num_rows)
    noise = rng.normal(scale=0.1, size=(num_rows, dim))
    
    vectors = centers[cluster_assignments] + noise
    vectors = vectors.astype(np.float32)
    return vectors

def get_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"bench_{run_id}"

def setup_lancedb():
    db = lancedb.connect("/home/user/myproject/lancedb")
    table_name = get_table_name()
    
    vectors = generate_data()
    data = pa.Table.from_arrays(
        [
            pa.array(np.arange(len(vectors))),
            pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1)), 384
            )
        ],
        names=["id", "vector"]
    )
    
    table = db.create_table(table_name, data, mode="overwrite")
    
    # Create IVF_PQ index
    table.create_index(
        metric="L2",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=32,
        num_sub_vectors=12,
        replace=True
    )
    
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=60))
    
    return table

def evaluate_recall(num_queries: int = 50, k: int = 10) -> float:
    db = lancedb.connect("/home/user/myproject/lancedb")
    table_name = get_table_name()
    table = db.open_table(table_name)
    
    rng = np.random.default_rng(999)
    queries = rng.normal(size=(num_queries, 384)).astype(np.float32)
    
    all_data = table.to_pandas()
    all_vectors = np.stack(all_data['vector'].values)
    all_ids = all_data['id'].values
    
    total_recall = 0.0
    for q in queries:
        dists = np.linalg.norm(all_vectors - q, axis=1)
        gt_indices = np.argsort(dists)[:k]
        gt_ids = set(all_ids[gt_indices])
        
        results = table.search(q).nprobes(10).refine_factor(20).limit(k).to_pandas()
        lancedb_ids = set(results['id'].values)
        
        recall = len(lancedb_ids.intersection(gt_ids)) / k
        total_recall += recall
        
    return total_recall / num_queries

if __name__ == "__main__":
    setup_lancedb()
    print(evaluate_recall())
