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
RUN mkdir -p /home/user/myproject/lancedb && chmod -R 0755 /home/user/myproject
RUN mkdir -p /app && chmod 0755 /app

# Seed the source LanceDB `articles` table at build time. 600 rows of
# (id, title, embedding[128]). The vectors are constructed so that their
# intrinsic dimensionality is exactly 16: a 16-d latent signal (with 3
# baked-in cluster offsets along three orthogonal latent axes) is embedded
# into 128-d via a random orthonormal projection, then perturbed with
# small isotropic noise. This guarantees PCA-16 captures ~99% of variance.
RUN python3 - <<'PYEOF'
import numpy as np
import pyarrow as pa
import lancedb

DB_PATH = "/home/user/myproject/lancedb"
N_PER_CLUSTER = 200
N_TOTAL = N_PER_CLUSTER * 3
LATENT = 16
DIM = 128
SEED = 2026

rng = np.random.default_rng(SEED)

# Three cluster means in the 16-d latent space, each offset along a
# different orthogonal latent axis.
means16 = np.zeros((3, LATENT), dtype=np.float64)
means16[0, 0] = 4.0
means16[1, 1] = 4.0
means16[2, 2] = 4.0

signal = np.zeros((N_TOTAL, LATENT), dtype=np.float64)
labels = np.zeros(N_TOTAL, dtype=np.int64)
for c in range(3):
    start = c * N_PER_CLUSTER
    end = start + N_PER_CLUSTER
    signal[start:end] = rng.standard_normal((N_PER_CLUSTER, LATENT)) + means16[c]
    labels[start:end] = c

# Random orthonormal 16->128 embedding (columns of Q are orthonormal).
A = rng.standard_normal((DIM, LATENT))
Q, _ = np.linalg.qr(A)  # Q shape (DIM, LATENT)
data = signal @ Q.T + 0.05 * rng.standard_normal((N_TOTAL, DIM))
vectors = data.astype(np.float32)

# Shuffle row order so IDs don't trivially correlate with clusters.
perm = rng.permutation(N_TOTAL)
vectors = vectors[perm]
labels = labels[perm]

ids = np.arange(N_TOTAL, dtype=np.int64)
titles = [f"article-{int(i):04d}-cluster{int(labels[i])}" for i in range(N_TOTAL)]

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("title", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), DIM)),
])

table_data = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "title": pa.array(titles, type=pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1).tolist(), type=pa.float32()), DIM
        ),
    },
    schema=schema,
)

db = lancedb.connect(DB_PATH)
if "articles" in db.table_names():
    db.drop_table("articles")
tbl = db.create_table("articles", data=table_data, mode="overwrite")
assert tbl.count_rows() == N_TOTAL, tbl.count_rows()
print(f"seeded source 'articles' table with {tbl.count_rows()} rows of {DIM}-d float32 vectors")

# lancedb 0.25.3 occasionally SIGABRTs at interpreter teardown after a
# write-heavy session; exit immediately to keep the build deterministic.
import os
os._exit(0)
PYEOF

# Sanity check: read the seed back and verify shape.
RUN python3 - <<'PYEOF'
import os
import lancedb
db = lancedb.connect("/home/user/myproject/lancedb")
assert "articles" in db.table_names(), db.table_names()
tbl = db.open_table("articles")
n = tbl.count_rows()
assert n == 600, n
schema = tbl.schema
field_names = [f.name for f in schema]
assert {"id", "title", "embedding"}.issubset(field_names), field_names
emb_field = schema.field("embedding")
assert emb_field.type.list_size == 128, emb_field.type
print("seed sanity OK: 600 rows, embedding fixed_size_list<float, 128>")
os._exit(0)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB PCA projection task. Source 'articles' table of 600 rows × 128-d float32 vectors baked into /home/user/myproject/lancedb at build time."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4
#           pandas==2.2.3 scikit-learn==1.5.2
#   - /home/user/myproject                    (project dir for the candidate)
#   - /home/user/myproject/lancedb/articles   (read-only seed: 600 × 128-d float32 vectors)
