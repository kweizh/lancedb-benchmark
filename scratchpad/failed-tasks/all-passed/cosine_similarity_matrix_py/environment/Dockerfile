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
RUN mkdir -p /home/user/myproject

ENV LANCEDB_URI=/home/user/myproject/lancedb_data
ENV LANCEDB_TABLE=vectors

# Seed the 200-row, 5-cluster, 48-dim cosine table at build time.
RUN python3 - <<'PYEOF'
import os
import numpy as np
import pyarrow as pa
import lancedb

URI = os.environ["LANCEDB_URI"]
TABLE = os.environ["LANCEDB_TABLE"]

N_PER_CLASS = 40
N_CLASSES = 5
N = N_PER_CLASS * N_CLASSES
DIM = 48
SEED = 2026

rng = np.random.default_rng(SEED)

# 5 well-separated cluster centres on orthogonal axes of the 48-d space.
centres = np.zeros((N_CLASSES, DIM), dtype=np.float32)
for c in range(N_CLASSES):
    # Each cluster sits on its own coordinate axis with magnitude 8.
    centres[c, c] = 8.0

ids = np.arange(N, dtype=np.int64)
labels = np.repeat(np.arange(N_CLASSES, dtype=np.int32), N_PER_CLASS)

# Each cluster: centre + small Gaussian noise (sigma 0.5) keeps clusters tight.
noise = rng.standard_normal(size=(N, DIM)).astype(np.float32) * 0.5
vectors = centres[labels] + noise

# Build-time invariant: intra-cluster cosine similarity should be much higher
# than inter-cluster similarity, so the candidate's matrix will obviously
# preserve cluster structure for >=4 of 5 labels.
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

intra = []
inter = []
for c in range(N_CLASSES):
    idx = np.where(labels == c)[0]
    for i in idx[:3]:
        for j in idx[3:6]:
            intra.append(_cos(vectors[i], vectors[j]))
    other = np.where(labels != c)[0]
    for i in idx[:3]:
        for j in other[:3]:
            inter.append(_cos(vectors[i], vectors[j]))
mean_intra = float(np.mean(intra))
mean_inter = float(np.mean(inter))
print(f"seed-check mean_intra_cos={mean_intra:.4f} mean_inter_cos={mean_inter:.4f}")
assert mean_intra > mean_inter + 0.5, (
    f"seed too noisy: intra={mean_intra} inter={mean_inter}"
)

vector_field = pa.field("vector", pa.list_(pa.float32(), DIM))
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("label", pa.int32()),
    vector_field,
])

records = []
for i in range(N):
    records.append({
        "id": int(ids[i]),
        "label": int(labels[i]),
        "vector": vectors[i].tolist(),
    })

db = lancedb.connect(URI)
if TABLE in db.table_names():
    db.drop_table(TABLE)
tbl = db.create_table(TABLE, data=records, schema=schema, mode="overwrite")
assert tbl.count_rows() == N, f"expected {N} rows, got {tbl.count_rows()}"
print(f"seeded {TABLE} with {tbl.count_rows()} rows at {URI}")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB cosine self-search matrix task: 200-row, 5-cluster, 48-d table seeded at build time."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: empty project directory awaiting candidate's solution.py
#   - /home/user/myproject/lancedb_data: pre-seeded LanceDB database with table `vectors`
#     (200 rows, ids 0..199, 5 clusters of 40 with int32 label, 48-d float32 vector)
#   - ENV LANCEDB_URI and LANCEDB_TABLE point the candidate at the seeded table
