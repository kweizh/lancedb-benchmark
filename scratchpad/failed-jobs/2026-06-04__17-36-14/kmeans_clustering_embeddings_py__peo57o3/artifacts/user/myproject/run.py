#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import lancedb
from sklearn.cluster import KMeans

def main():
    db_path = "/home/user/myproject/lancedb_data"
    db = lancedb.connect(db_path)
    
    # 1. Load all rows of the embeddings table into memory as a pandas DataFrame.
    tbl = db.open_table("embeddings")
    df = tbl.to_pandas()
    
    # 2. Stack the vector column into a 2D numpy array and fit KMeans.
    vectors = np.vstack(df["vector"].values)
    kmeans = KMeans(n_clusters=8, random_state=2026, n_init=10)
    kmeans.fit(vectors)
    
    # 3. Persist the cluster assignments into 'clusters' table.
    # Columns: id (Int64), cluster_id (Int32).
    clusters_df = pd.DataFrame({
        "id": df["id"].astype(np.int64),
        "cluster_id": kmeans.labels_.astype(np.int32)
    })
    clusters_schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("cluster_id", pa.int32())
    ])
    clusters_table = pa.Table.from_pandas(clusters_df, schema=clusters_schema, preserve_index=False)
    db.create_table("clusters", data=clusters_table, mode="overwrite")
    
    # 4. Persist the 8 centroids into 'centroids' table.
    # Columns: cluster_id (Int32), vector (fixed-size list of float32, dimension 32).
    # row i = centroid for cluster_id i.
    centroids_data = []
    for i, centroid in enumerate(kmeans.cluster_centers_):
        centroids_data.append({
            "cluster_id": i,
            "vector": centroid.astype(np.float32).tolist()
        })
    centroids_schema = pa.schema([
        pa.field("cluster_id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 32))
    ])
    centroids_table = pa.Table.from_pylist(centroids_data, schema=centroids_schema)
    db.create_table("centroids", data=centroids_table, mode="overwrite")
    
    print("Clustering completed and tables persisted successfully.")

if __name__ == "__main__":
    main()
