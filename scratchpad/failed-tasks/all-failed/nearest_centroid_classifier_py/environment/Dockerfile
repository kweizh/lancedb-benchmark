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
RUN mkdir -p /app/lancedb_data

# Seed script: deterministic 600-row training table with 6 well-separated
# clusters in 40-d (orthonormal class axes scaled by 5.0, Gaussian noise
# sigma=0.5). Also writes the 120-row test fixture to /app/test_set.json.
# Build time uses ZEALT_RUN_ID="build"; the entrypoint re-seeds under the real
# ZEALT_RUN_ID before the task starts.
RUN cat >/app/seed.py <<'PYEOF'
import json
import os

import numpy as np
import pyarrow as pa
import lancedb

RUN_ID = os.environ.get("ZEALT_RUN_ID", "build")
LANCEDB_DIR = "/app/lancedb_data"
TRAIN_TABLE = f"train_data_{RUN_ID}"
CENTROIDS_TABLE = f"centroids_{RUN_ID}"
TEST_SET_PATH = "/app/test_set.json"

SEED = 2026
DIM = 40
NUM_CLASSES = 6
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 20
CENTER_SCALE = 5.0
NOISE_SIGMA = 0.5

rng = np.random.default_rng(SEED)

# Generate NUM_CLASSES orthonormal class axes in 40-d via QR decomposition.
A = rng.standard_normal((DIM, NUM_CLASSES)).astype(np.float64)
Q, _ = np.linalg.qr(A)  # Q has orthonormal columns of shape (DIM, NUM_CLASSES)
centers = (CENTER_SCALE * Q.T).astype(np.float64)  # shape (NUM_CLASSES, DIM)

# Sanity check: pairwise center distances are all large (well-separated).
for a in range(NUM_CLASSES):
    for b in range(a + 1, NUM_CLASSES):
        d = float(np.linalg.norm(centers[a] - centers[b]))
        # Orthogonal unit vectors scaled by 5 are sqrt(2)*5 ~ 7.07 apart.
        assert d > 6.0, f"Centers {a},{b} not well-separated: {d}"

# Build training rows: 100 per class.
train_rows = []
row_id = 0
for c in range(NUM_CLASSES):
    cls_noise = rng.standard_normal((TRAIN_PER_CLASS, DIM)).astype(np.float64) * NOISE_SIGMA
    cls_vecs = centers[c] + cls_noise
    for v in cls_vecs:
        train_rows.append({
            "id": int(row_id),
            "label": int(c),
            "vector": v.astype(np.float32).tolist(),
        })
        row_id += 1

# Build test rows: 20 per class, independent draws.
test_rows = []
for c in range(NUM_CLASSES):
    cls_noise = rng.standard_normal((TEST_PER_CLASS, DIM)).astype(np.float64) * NOISE_SIGMA
    cls_vecs = centers[c] + cls_noise
    for v in cls_vecs:
        test_rows.append({
            "vector": v.astype(np.float32).tolist(),
            "label": int(c),
        })

db = lancedb.connect(LANCEDB_DIR)

# Drop any pre-existing tables under this run id so seeding is idempotent.
existing = db.table_names()
if TRAIN_TABLE in existing:
    db.drop_table(TRAIN_TABLE)
if CENTROIDS_TABLE in existing:
    db.drop_table(CENTROIDS_TABLE)

train_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("label", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])
arrow_tbl = pa.Table.from_pylist(train_rows, schema=train_schema)
tbl = db.create_table(TRAIN_TABLE, data=arrow_tbl)
assert tbl.count_rows() == NUM_CLASSES * TRAIN_PER_CLASS, (
    f"Expected {NUM_CLASSES * TRAIN_PER_CLASS} rows, got {tbl.count_rows()}"
)

# Persist the test fixture.
with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
    json.dump(test_rows, f)

# Workaround lancedb 0.25.3 interpreter-teardown SIGABRT.
os._exit(0)
PYEOF

RUN ZEALT_RUN_ID=build python3 /app/seed.py

# Entrypoint re-seeds the training table + test fixture under the real
# ZEALT_RUN_ID so parallel runs don't collide on table names.
RUN cat >/app/entrypoint.sh <<'SHEOF'
#!/bin/bash
set -u
if [ -n "${ZEALT_RUN_ID:-}" ] && [ "${ZEALT_RUN_ID}" != "build" ]; then
  python3 /app/seed.py >/app/seed.log 2>&1 || true
fi
exec "$@"
SHEOF
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB nearest-centroid classifier task with a 600-row training table and a 120-row test fixture."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /app/seed.py builds the 600-row train_data_${ZEALT_RUN_ID} LanceDB table (6 classes,
#     100 rows/class, 40-d float32 vectors) and writes /app/test_set.json (120 rows).
#   - /app/entrypoint.sh re-seeds tables under the real ZEALT_RUN_ID before the task starts.
#   - /home/user/myproject is the empty project directory where the candidate writes
#     solution.py (build_centroids / classify / evaluate) and run.py.
