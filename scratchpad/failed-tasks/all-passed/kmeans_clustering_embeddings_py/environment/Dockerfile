FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        scikit-learn==1.5.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject/lancedb_data

# Seed the LanceDB `embeddings` table and ground-truth labels at build time.
# 800 rows = 8 clusters of 100, centers along the first 8 standard-basis axes,
# magnitude 15, Gaussian noise std=1. Trivially separable so KMeans recovers ARI~1.
RUN cat > /tmp/_seed.py <<'PYEOF' && python3 /tmp/_seed.py && rm -f /tmp/_seed.py
import os
import numpy as np
import pyarrow as pa
import lancedb

LANCEDB_DIR = "/home/user/myproject/lancedb_data"
DIM = 32
CLUSTERS = 8
PER_CLUSTER = 100
SEED = 2026
CENTER_SCALE = 15.0
NOISE_STD = 1.0

os.makedirs(LANCEDB_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

centers = np.zeros((CLUSTERS, DIM), dtype=np.float32)
for c in range(CLUSTERS):
    centers[c, c] = CENTER_SCALE

vectors = np.empty((CLUSTERS * PER_CLUSTER, DIM), dtype=np.float32)
labels = np.empty(CLUSTERS * PER_CLUSTER, dtype=np.int64)
for c in range(CLUSTERS):
    start = c * PER_CLUSTER
    end = start + PER_CLUSTER
    noise = rng.standard_normal((PER_CLUSTER, DIM)).astype(np.float32) * NOISE_STD
    vectors[start:end] = centers[c] + noise
    labels[start:end] = c

ids = np.arange(vectors.shape[0], dtype=np.int64)
np.save(os.path.join(LANCEDB_DIR, "ground_truth.npy"), labels)

vec_type = pa.list_(pa.float32(), DIM)
schema = pa.schema([pa.field("id", pa.int64()), pa.field("vector", vec_type)])
arr_id = pa.array(ids, type=pa.int64())
arr_vec = pa.FixedSizeListArray.from_arrays(
    pa.array(vectors.reshape(-1).tolist(), type=pa.float32()), DIM
)
table = pa.Table.from_arrays([arr_id, arr_vec], schema=schema)

db = lancedb.connect(LANCEDB_DIR)
if "embeddings" in db.table_names():
    db.drop_table("embeddings")
db.create_table("embeddings", data=table)

reopened = db.open_table("embeddings")
assert reopened.count_rows() == CLUSTERS * PER_CLUSTER, "row count drift"
field_names = {f.name for f in reopened.schema}
assert field_names == {"id", "vector"}, f"schema drift: {field_names}"

from numpy.linalg import norm
inter = norm(centers[0] - centers[1])
intra_max = 0.0
for c in range(CLUSTERS):
    start = c * PER_CLUSTER
    end = start + PER_CLUSTER
    intra_max = max(intra_max, float(np.max(norm(vectors[start:end] - centers[c], axis=1))))
assert intra_max < inter, f"separability fail: intra_max={intra_max:.3f} inter={inter:.3f}"

print(f"_seed.py: wrote {vectors.shape[0]} rows; intra_max={intra_max:.3f} inter={inter:.3f}")
del reopened, db
os._exit(0)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 + scikit-learn 1.5.2 K-means clustering task; pre-seeded 800-row 32-d embeddings table with 8 ground-truth clusters."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, scikit-learn==1.5.2
#   - /home/user/myproject: project directory the candidate works in
#   - /home/user/myproject/lancedb_data: pre-seeded LanceDB directory containing
#     the `embeddings` table (800 rows, 32-d vectors, 8 ground-truth clusters) and
#     `ground_truth.npy` (per-row labels in id-sorted order).
