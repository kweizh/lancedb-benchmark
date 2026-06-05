import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

def main():
    db_path = "/home/user/myproject/lancedb_data"
    db = lancedb.connect(db_path)
    
    # Read embeddings
    embeddings_table = db.open_table("embeddings")
    df = embeddings_table.to_pandas()
    
    X = np.vstack(df['vector'].values)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=8, random_state=2026, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Prepare clusters table
    clusters_df = pd.DataFrame({
        "id": df["id"].astype("int64"),
        "cluster_id": cluster_labels.astype("int32")
    })
    
    # Overwrite clusters table
    db.create_table("clusters", data=clusters_df, mode="overwrite")
    
    # Prepare centroids table
    centroids = kmeans.cluster_centers_
    centroids_df = pd.DataFrame({
        "cluster_id": np.arange(8, dtype="int32"),
        "vector": list(centroids.astype("float32"))
    })
    
    schema = pa.schema([
        pa.field("cluster_id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 32))
    ])
    
    db.create_table("centroids", data=centroids_df, schema=schema, mode="overwrite")
    
    os._exit(0)

if __name__ == "__main__":
    main()
