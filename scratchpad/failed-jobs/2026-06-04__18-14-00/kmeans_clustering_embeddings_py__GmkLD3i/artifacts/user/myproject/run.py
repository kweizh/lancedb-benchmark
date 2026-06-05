"""
run.py – Fit K-means on the LanceDB `embeddings` table and persist
         the results to `clusters` and `centroids` LanceDB tables.
"""

import numpy as np
import pandas as pd
import lancedb
import pyarrow as pa
from sklearn.cluster import KMeans

LANCEDB_PATH = "/home/user/myproject/lancedb_data"
N_CLUSTERS = 8
RANDOM_STATE = 2026
N_INIT = 10
VECTOR_DIM = 32


def main():
    # ------------------------------------------------------------------ #
    # 1. Connect and load embeddings
    # ------------------------------------------------------------------ #
    db = lancedb.connect(LANCEDB_PATH)
    emb_table = db.open_table("embeddings")
    df = emb_table.to_pandas()

    # Stack individual per-row numpy arrays into a (800, 32) matrix
    X = np.vstack(df["vector"].values).astype(np.float32)
    ids = df["id"].values  # keep original ids for the clusters table

    # ------------------------------------------------------------------ #
    # 2. Fit K-means (fully deterministic)
    # ------------------------------------------------------------------ #
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    kmeans.fit(X)

    labels = kmeans.labels_.astype(np.int32)          # shape (800,)
    centroids = kmeans.cluster_centers_.astype(np.float32)  # shape (8, 32)

    # ------------------------------------------------------------------ #
    # 3. Write `clusters` table  (overwrite if it already exists)
    # ------------------------------------------------------------------ #
    clusters_schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("cluster_id", pa.int32()),
    ])

    clusters_df = pd.DataFrame({
        "id": ids.astype(np.int64),
        "cluster_id": labels,
    })

    clusters_table_data = pa.Table.from_pandas(clusters_df, schema=clusters_schema, preserve_index=False)

    if "clusters" in db.table_names():
        db.drop_table("clusters")
    db.create_table("clusters", data=clusters_table_data)

    print(f"[run.py] Written 'clusters' table: {len(clusters_df)} rows")

    # ------------------------------------------------------------------ #
    # 4. Write `centroids` table  (overwrite if it already exists)
    # ------------------------------------------------------------------ #
    # LanceDB expects fixed-size list vectors as a list of lists / arrays
    centroids_schema = pa.schema([
        pa.field("cluster_id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])

    centroids_rows = [
        {"cluster_id": np.int32(i), "vector": centroids[i].tolist()}
        for i in range(N_CLUSTERS)
    ]
    centroids_table_data = pa.Table.from_pylist(centroids_rows, schema=centroids_schema)

    if "centroids" in db.table_names():
        db.drop_table("centroids")
    db.create_table("centroids", data=centroids_table_data)

    print(f"[run.py] Written 'centroids' table: {N_CLUSTERS} rows")
    print("[run.py] Done.")


if __name__ == "__main__":
    main()
