#!/usr/bin/env python3
"""Read embeddings from LanceDB, fit KMeans, and persist clusters + centroids tables."""

import numpy as np
import pandas as pd
import lancedb
from sklearn.cluster import KMeans

DB_PATH = "/home/user/myproject/lancedb_data"


def main():
    db = lancedb.connect(DB_PATH)

    # --- Load embeddings ---
    emb_tbl = db.open_table("embeddings")
    df = emb_tbl.to_pandas()
    ids = df["id"].values  # int64
    vectors = np.vstack(df["vector"].values).astype(np.float32)

    # --- Fit KMeans ---
    kmeans = KMeans(n_clusters=8, random_state=2026, n_init=10)
    kmeans.fit(vectors)

    labels = kmeans.labels_  # shape (800,), values 0..7
    centroids = kmeans.cluster_centers_  # shape (8, 32), float64 by default

    # --- Persist clusters table ---
    clusters_df = pd.DataFrame({
        "id": ids,
        "cluster_id": labels.astype(np.int32),
    })

    try:
        db.drop_table("clusters")
    except Exception:
        pass
    db.create_table("clusters", clusters_df)

    # --- Persist centroids table ---
    # Build a list-of-lists for the vector column; dtype float32
    centroids_list = centroids.astype(np.float32).tolist()
    centroids_df = pd.DataFrame({
        "cluster_id": np.arange(8, dtype=np.int32),
        "vector": centroids_list,
    })

    try:
        db.drop_table("centroids")
    except Exception:
        pass
    db.create_table("centroids", centroids_df)

    print(f"Clusters table: {len(clusters_df)} rows")
    print(f"Centroids table: {len(centroids_df)} rows")
    print("Done.")


if __name__ == "__main__":
    main()