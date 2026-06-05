#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import lancedb
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def run_checks():
    print("--- Starting Verification ---")
    
    # 1. Run run.py to ensure it is idempotent and runs successfully
    print("Running run.py...")
    ret = os.system("python3 /home/user/myproject/run.py")
    if ret != 0:
        print("ERROR: run.py failed with exit code", ret)
        sys.exit(1)
    print("run.py executed successfully.")
    
    # 2. Check lancedb tables
    db_path = "/home/user/myproject/lancedb_data"
    db = lancedb.connect(db_path)
    
    # Check embeddings table
    if "embeddings" not in db.table_names():
        print("ERROR: embeddings table not found")
        sys.exit(1)
    emb_tbl = db.open_table("embeddings")
    emb_df = emb_tbl.to_pandas()
    
    # Check clusters table
    if "clusters" not in db.table_names():
        print("ERROR: clusters table not found")
        sys.exit(1)
    clusters_tbl = db.open_table("clusters")
    clusters_df = clusters_tbl.to_pandas()
    
    # Check clusters schema
    schema = clusters_tbl.schema
    print("Clusters schema:", schema)
    # Check column types
    # id: int64, cluster_id: int32
    id_field = schema.field("id")
    cluster_id_field = schema.field("cluster_id")
    if id_field.type != "int64":
        print(f"ERROR: 'id' column type is {id_field.type}, expected int64")
        sys.exit(1)
    if cluster_id_field.type != "int32":
        print(f"ERROR: 'cluster_id' column type is {cluster_id_field.type}, expected int32")
        sys.exit(1)
    
    # Check row counts
    if len(clusters_df) != 800:
        print(f"ERROR: clusters table has {len(clusters_df)} rows, expected 800")
        sys.exit(1)
    
    # Check set of id values
    if set(clusters_df["id"]) != set(emb_df["id"]):
        print("ERROR: set of ids in clusters table does not match embeddings table")
        sys.exit(1)
        
    # Check centroids table
    if "centroids" not in db.table_names():
        print("ERROR: centroids table not found")
        sys.exit(1)
    centroids_tbl = db.open_table("centroids")
    centroids_df = centroids_tbl.to_pandas()
    
    # Check centroids schema
    c_schema = centroids_tbl.schema
    print("Centroids schema:", c_schema)
    c_id_field = c_schema.field("cluster_id")
    vector_field = c_schema.field("vector")
    if c_id_field.type != "int32":
        print(f"ERROR: centroids 'cluster_id' column type is {c_id_field.type}, expected int32")
        sys.exit(1)
    # Check if vector is fixed_size_list of float32, length 32
    # In pyarrow, fixed_size_list type has list_size and value_type attributes
    import pyarrow as pa
    if not isinstance(vector_field.type, pa.FixedSizeListType):
        print(f"ERROR: vector column is not a FixedSizeListType: {vector_field.type}")
        sys.exit(1)
    if vector_field.type.list_size != 32:
        print(f"ERROR: vector list size is {vector_field.type.list_size}, expected 32")
        sys.exit(1)
    if vector_field.type.value_type != "float":
        print(f"ERROR: vector value type is {vector_field.type.value_type}, expected float32")
        sys.exit(1)
        
    if len(centroids_df) != 8:
        print(f"ERROR: centroids table has {len(centroids_df)} rows, expected 8")
        sys.exit(1)
        
    # 3. Check cluster quality
    # The 8 distinct cluster ids {0..7} must all be present
    unique_clusters = set(clusters_df["cluster_id"])
    if unique_clusters != set(range(8)):
        print(f"ERROR: unique cluster ids are {unique_clusters}, expected {set(range(8))}")
        sys.exit(1)
        
    # Check balance (80 to 120 inclusive)
    counts = clusters_df["cluster_id"].value_counts()
    print("Cluster counts:\n", counts)
    for cid, count in counts.items():
        if count < 80 or count > 120:
            print(f"ERROR: Cluster {cid} has {count} elements, which is outside [80, 120]")
            sys.exit(1)
            
    # Check ARI against ground truth
    gt = np.load("/home/user/myproject/lancedb_data/ground_truth.npy")
    # Note: clusters_df is not necessarily sorted by id. Let's merge or sort to align with gt
    merged = emb_df[["id"]].merge(clusters_df, on="id")
    predicted_labels = merged["cluster_id"].values
    ari = adjusted_rand_score(gt, predicted_labels)
    print("Adjusted Rand Index (ARI):", ari)
    if ari < 0.90:
        print("ERROR: ARI is less than 0.90:", ari)
        sys.exit(1)
        
    # 4. Import solution and verify top-level callables
    print("Importing solution...")
    sys.path.insert(0, "/home/user/myproject")
    try:
        import solution
    except Exception as e:
        print("ERROR: Failed to import solution:", e)
        sys.exit(1)
        
    # Check cluster_centroids()
    print("Checking solution.cluster_centroids()...")
    centroids = solution.cluster_centroids()
    if not isinstance(centroids, np.ndarray):
        print("ERROR: cluster_centroids() did not return a numpy array")
        sys.exit(1)
    if centroids.shape != (8, 32):
        print(f"ERROR: cluster_centroids() returned shape {centroids.shape}, expected (8, 32)")
        sys.exit(1)
    if centroids.dtype != np.float32:
        print(f"ERROR: cluster_centroids() returned dtype {centroids.dtype}, expected float32")
        sys.exit(1)
        
    # Re-run KMeans to compare centroids
    vectors = np.vstack(emb_df["vector"].values).astype(np.float64)
    kmeans = KMeans(n_clusters=8, random_state=2026, n_init=10).fit(vectors)
    expected_centroids = kmeans.cluster_centers_.astype(np.float32)
    
    # Sort expected centroids by cluster_id if not already (they are in cluster_id order by default in sklearn, i.e. cluster i centroid is expected_centroids[i])
    # Let's verify closeness
    if not np.allclose(centroids, expected_centroids, atol=1e-4):
        print("ERROR: Centroids do not match expected kmeans.cluster_centers_ within atol=1e-4")
        # Print differences
        diff = np.abs(centroids - expected_centroids)
        print("Max absolute difference:", np.max(diff))
        sys.exit(1)
    print("Centroids match perfectly!")
    
    # 5. Check nearest_cluster with 5 deterministic query vectors from the dataset
    print("Checking solution.nearest_cluster()...")
    # Let's select 5 deterministic query vectors from the dataset that span different clusters
    labels = kmeans.labels_
    indices = []
    for i in range(5):
        # find first index belonging to cluster i
        idx = int(np.where(labels == i)[0][0])
        indices.append(idx)
    queries = vectors[indices]
    
    for idx, query in enumerate(queries):
        pred_kmeans = kmeans.predict([query.tolist()])[0]
        pred_nearest = solution.nearest_cluster(query.tolist())
        print(f"Query {idx} (from dataset index {indices[idx]}): KMeans predicts={pred_kmeans}, nearest_cluster={pred_nearest}")
        if pred_nearest != pred_kmeans:
            print(f"ERROR: nearest_cluster returned {pred_nearest}, expected {pred_kmeans}")
            sys.exit(1)
            
    print("--- Verification Successful! ---")

if __name__ == "__main__":
    run_checks()
