import os
import datetime
import lancedb
import numpy as np
import pyarrow as pa

# Deterministic seeds
SEED_CENTERS = 42
SEED_DATA = 100
SEED_QUERIES = 200

def get_centers(num_clusters=10, dim=384):
    rng = np.random.default_rng(SEED_CENTERS)
    centers = rng.normal(size=(num_clusters, dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    return centers

def generate_table_vectors(num_rows=1024, dim=384):
    centers = get_centers(10, dim)
    rng = np.random.default_rng(SEED_DATA)
    vectors = []
    for i in range(num_rows):
        cluster_idx = i % 10
        center = centers[cluster_idx]
        noise = rng.normal(scale=0.05, size=dim)
        vector = center + noise
        vectors.append(vector.astype(np.float32))
    return np.array(vectors)

def generate_query_vectors(num_queries=50, dim=384):
    centers = get_centers(10, dim)
    rng = np.random.default_rng(SEED_QUERIES)
    queries = []
    for i in range(num_queries):
        cluster_idx = i % 10
        center = centers[cluster_idx]
        noise = rng.normal(scale=0.05, size=dim)
        query = center + noise
        queries.append(query.astype(np.float32))
    return np.array(queries)

def setup_db_and_index():
    db_dir = "/home/user/myproject/lancedb"
    os.makedirs(db_dir, exist_ok=True)
    
    db = lancedb.connect(db_dir)
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"bench_{run_id}"
    
    # Recreate table from scratch (idempotent)
    if table_name in db.table_names():
        db.drop_table(table_name)
        
    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 384))
    ])
    
    vectors = generate_table_vectors(1024, 384)
    data = []
    for idx, vec in enumerate(vectors):
        data.append({
            "id": idx,
            "vector": vec.tolist()
        })
        
    table = db.create_table(table_name, data=data, schema=schema, mode="overwrite")
    
    # Create IVF_PQ index with aggressively low num_sub_vectors (e.g. 12)
    # We choose num_partitions=16
    table.create_index(
        metric="L2",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=16,
        num_sub_vectors=8,
        replace=True,
        name="vector_idx"
    )
    
    table.wait_for_index(index_names=["vector_idx"], timeout=datetime.timedelta(seconds=60))
    return table

def evaluate_recall(num_queries: int = 50, k: int = 10) -> float:
    db_dir = "/home/user/myproject/lancedb"
    db = lancedb.connect(db_dir)
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"bench_{run_id}"
    table = db.open_table(table_name)
    
    # 1. Deterministically draw query vectors
    query_vectors = generate_query_vectors(num_queries, 384)
    
    # 2. Read every vector back from the table for brute-force ground truth
    df = table.to_pandas()
    ids = df["id"].values
    table_vectors = np.stack(df["vector"].values)
    
    total_recall = 0.0
    
    # Search parameters
    nprobes = 16          # Probe all partitions for maximum partition coverage
    refine_factor = 20    # Re-rank top candidates with exact FP32 distance to recover recall
    
    for i in range(num_queries):
        qv = query_vectors[i]
        
        # Compute brute-force ground truth L2 distance
        dists = np.sum((table_vectors - qv) ** 2, axis=1)
        ground_truth_indices = np.argsort(dists)[:k]
        ground_truth_ids = set(ids[ground_truth_indices])
        
        # Run search through LanceDB IVF_PQ index
        results = (
            table.search(qv)
            .metric("L2")
            .nprobes(nprobes)
            .refine_factor(refine_factor)
            .limit(k)
            .to_pandas()
        )
        returned_ids = set(results["id"].values)
        
        # Compute recall@k
        intersection = returned_ids.intersection(ground_truth_ids)
        recall_i = len(intersection) / k
        total_recall += recall_i
        
    avg_recall = total_recall / num_queries
    return avg_recall

if __name__ == "__main__":
    setup_db_and_index()
    recall = evaluate_recall()
    print(f"Recall: {recall:.4f}")
