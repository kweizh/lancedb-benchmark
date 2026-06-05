FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject

# Bake the seed script. It runs at container start (entrypoint) because the
# table name depends on ZEALT_RUN_ID, which is only available at runtime.
RUN cat >/home/user/myproject/_seed.py <<'PYEOF'
"""Seed the LanceDB `documents_${ZEALT_RUN_ID}` table with a rigged 400-row,
4-cluster 32-d fixture. Designed so a single Rocchio update flips top-1
from cluster 1 to cluster 0 (doc 55 = the cluster-0 centroid)."""
import json
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

run_id = os.environ.get("ZEALT_RUN_ID", "")
if not run_id:
    print("[seed] ERROR: ZEALT_RUN_ID is required.", file=sys.stderr)
    sys.exit(1)

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
QUERY_PATH = os.path.join(PROJECT_DIR, "query.json")
TABLE_NAME = f"documents_{run_id}"
MARKER = os.path.join(PROJECT_DIR, f".seeded_{run_id}")

if os.path.exists(MARKER):
    print(f"[seed] Already seeded ({MARKER}), skipping.", flush=True)
    os._exit(0)

os.makedirs(DB_DIR, exist_ok=True)

DIM = 32
N_CLUSTERS = 4
N_PER_CLUSTER = 100
N_TOTAL = N_CLUSTERS * N_PER_CLUSTER
NOISE_SCALE = 0.05
SEED = 2026

rng = np.random.default_rng(SEED)

# Cluster centers: e_c (one-hot in dimension c) for c = 0..3.
centers = np.zeros((N_CLUSTERS, DIM), dtype=np.float64)
for c in range(N_CLUSTERS):
    centers[c, c] = 1.0

vectors = np.zeros((N_TOTAL, DIM), dtype=np.float64)
clusters = np.zeros(N_TOTAL, dtype=np.int64)
ids = np.arange(N_TOTAL, dtype=np.int64)

# Each doc = (cluster_center + noise) / ||...||, with noise restricted to
# dims [N_CLUSTERS .. DIM) so it's orthogonal to every cluster center and
# to q0. This keeps inter-cluster cosines clean and predictable.
for c in range(N_CLUSTERS):
    base = c * N_PER_CLUSTER
    for j in range(N_PER_CLUSTER):
        i = base + j
        clusters[i] = c
        v = centers[c].copy()
        noise_tail = rng.standard_normal(DIM - N_CLUSTERS) * NOISE_SCALE
        v[N_CLUSTERS:] = noise_tail
        v = v / np.linalg.norm(v)
        vectors[i] = v

# Doc 55 (cluster 0) is rigged to sit EXACTLY at the cluster-0 centroid.
vectors[55] = centers[0].copy()

# q0 = (0.4*e0 + 0.6*e1) normalized -- biased toward cluster 1, so the
# initial top-10 lands entirely in cluster 1. The user-provided relevant
# ids {50..54} are NOT in the initial top-10, which means the candidate
# must fetch their vectors directly from the table.
q0_raw = 0.4 * centers[0] + 0.6 * centers[1]
q0 = q0_raw / np.linalg.norm(q0_raw)

# Verify all stored vectors are L2-normalized.
norms = np.linalg.norm(vectors, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-6), (
    f"All vectors must be L2-normalized. min={norms.min()}, max={norms.max()}"
)

# Simulate the candidate's Rocchio update and confirm the rigged invariants
# hold for THIS fixture. If the simulation fails, fail the build (rather
# than shipping a broken task).
sims_q0 = vectors @ q0
top10_initial = np.argsort(-sims_q0)[:10]
init_clusters = clusters[top10_initial]
assert np.all(init_clusters == 1), (
    f"Initial top-10 must all be cluster 1 (the 'wrong' cluster). "
    f"Got clusters: {init_clusters.tolist()}, ids: {top10_initial.tolist()}"
)

relevant_ids = [50, 51, 52, 53, 54]
rel_vecs = vectors[relevant_ids]
nrel_vecs = vectors[top10_initial]
alpha, beta, gamma = 1.0, 0.75, 0.15
q_prime_raw = (
    alpha * q0
    + beta * rel_vecs.mean(axis=0)
    - gamma * nrel_vecs.mean(axis=0)
)
q_prime = q_prime_raw / np.linalg.norm(q_prime_raw)

sims_q_prime = vectors @ q_prime
top10_after = np.argsort(-sims_q_prime)[:10]
after_clusters = clusters[top10_after]
print(
    f"[seed] After-Rocchio top-10 ids={top10_after.tolist()} "
    f"clusters={after_clusters.tolist()}"
)
assert int(top10_after[0]) == 55, (
    f"Rocchio top-1 must be doc 55 (cluster-0 centroid). Got {top10_after[0]}."
)
top5_cluster0 = int(np.sum((top10_after[:5] >= 0) & (top10_after[:5] < 100)))
assert top5_cluster0 >= 4, (
    f"Top-5 after Rocchio must contain ≥4 cluster-0 docs. Got {top5_cluster0}."
)

# Write the LanceDB table.
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("cluster", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

vectors_f32 = vectors.astype(np.float32)
table_data = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "cluster": pa.array(clusters, type=pa.int64()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors_f32.reshape(-1).tolist(), type=pa.float32()),
            DIM,
        ),
    },
    schema=schema,
)

db = lancedb.connect(DB_DIR)
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)
tbl = db.create_table(TABLE_NAME, data=table_data, mode="create")
assert tbl.count_rows() == N_TOTAL, tbl.count_rows()
print(f"[seed] Created '{TABLE_NAME}' with {tbl.count_rows()} rows.", flush=True)

# Persist the query fixture for the candidate's run.py.
with open(QUERY_PATH, "w") as f:
    json.dump(
        {"q0": q0.astype(np.float32).tolist(), "relevant_ids": relevant_ids},
        f,
    )
print(f"[seed] Wrote {QUERY_PATH}.", flush=True)

with open(MARKER, "w") as f:
    f.write("ok\n")
print(f"[seed] Wrote marker {MARKER}.", flush=True)

# lancedb 0.25.3 occasionally SIGABRTs at interpreter teardown after a
# write-heavy session; exit hard to keep entrypoint deterministic.
os._exit(0)
PYEOF

RUN cat >/home/user/myproject/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u
RUN_ID="${ZEALT_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
    echo "[entrypoint] ERROR: ZEALT_RUN_ID is not set." >&2
    exit 1
fi
MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "$MARKER" ]; then
    python3 /home/user/myproject/_seed.py || {
        echo "[entrypoint] Seed failed; exiting." >&2
        exit 1
    }
fi
trap : TERM INT
sleep infinity & wait
SHEOF

RUN chmod +x /home/user/myproject/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB Rocchio pseudo-relevance feedback task: 400-row, 4-cluster 32-d fixture rigged so a single Rocchio update flips top-1 from cluster 1 to doc 55 (cluster-0 centroid)."

CMD ["/home/user/myproject/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /home/user/myproject/_seed.py    seeds documents_${ZEALT_RUN_ID} at first start
#   - /home/user/myproject/entrypoint.sh  runs the seed (idempotent via marker) then sleeps
