"""
run.py – fit K-means on LanceDB embeddings and persist results.

Idempotent: drops and recreates `clusters` and `centroids` tables on every run.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import lancedb
from sklearn.cluster import KMeans

LANCEDB_DIR = "/home/user/myproject/lancedb_data"
N_CLUSTERS = 8
RANDOM_STATE = 2026
N_INIT = 10


def main():
    # ── 1. Load embeddings ────────────────────────────────────────────────────
    db = lancedb.connect(LANCEDB_DIR)
    emb_table = db.open_table("embeddings")
    df = emb_table.to_pandas()                    # columns: id, vector

    ids = df["id"].values                         # int64 array (800,)
    X = np.vstack(df["vector"].values).astype("float32")  # (800, 32)

    # ── 2. Fit K-means ────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    kmeans.fit(X)

    labels = kmeans.labels_.astype("int32")                   # (800,)
    centroids = kmeans.cluster_centers_.astype("float32")     # (8, 32)

    # ── 3. Persist `clusters` table ──────────────────────────────────────────
    clusters_schema = pa.schema([
        pa.field("id",         pa.int64()),
        pa.field("cluster_id", pa.int32()),
    ])
    clusters_table = pa.table(
        {
            "id":         pa.array(ids,    type=pa.int64()),
            "cluster_id": pa.array(labels, type=pa.int32()),
        },
        schema=clusters_schema,
    )

    if "clusters" in db.table_names():
        db.drop_table("clusters")
    db.create_table("clusters", data=clusters_table, schema=clusters_schema)
    print(f"clusters table written: {len(ids)} rows")

    # ── 4. Persist `centroids` table ─────────────────────────────────────────
    centroids_schema = pa.schema([
        pa.field("cluster_id", pa.int32()),
        pa.field("vector",     pa.list_(pa.float32(), 32)),
    ])
    centroids_table = pa.table(
        {
            "cluster_id": pa.array(np.arange(N_CLUSTERS, dtype="int32"), type=pa.int32()),
            "vector":     pa.FixedSizeListArray.from_arrays(
                              pa.array(centroids.flatten(), type=pa.float32()), 32
                          ),
        },
        schema=centroids_schema,
    )

    if "centroids" in db.table_names():
        db.drop_table("centroids")
    db.create_table("centroids", data=centroids_table, schema=centroids_schema)
    print(f"centroids table written: {N_CLUSTERS} rows")

    print("Done.")


if __name__ == "__main__":
    main()
