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

# Build-time fixture: deterministic 1024x64 dataset and 30x64 query vectors.
RUN mkdir -p /app/fixtures && \
    python3 - <<'PY'
import json
import os

import numpy as np

rng = np.random.default_rng(2026)

# 8 well-separated cluster centroids in 64-d so the brute-force top-10 is stable.
n_clusters = 8
dim = 64
n_rows = 1024
centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3.0

# Assign rows to clusters round-robin and add small noise.
labels = np.tile(np.arange(n_clusters), n_rows // n_clusters)
data = centroids[labels] + rng.standard_normal((n_rows, dim)).astype(np.float32) * 0.5

# 30 query vectors drawn from the same generative model so PQ recall is meaningful.
n_q = 30
q_labels = rng.integers(0, n_clusters, size=n_q)
queries = centroids[q_labels] + rng.standard_normal((n_q, dim)).astype(np.float32) * 0.5

np.save("/app/fixtures/data.npy", data.astype(np.float32))
np.save("/app/fixtures/queries.npy", queries.astype(np.float32))

with open("/app/fixtures/metadata.json", "w") as f:
    json.dump({"rows": n_rows, "dim": dim, "num_queries": n_q, "seed": 2026}, f)

print("fixture data shape:", data.shape, data.dtype)
print("fixture queries shape:", queries.shape, queries.dtype)
PY

# Build-time sanity check: verify that the PQ recall sweep is monotonic on
# this fixture using lancedb 0.25.3. If the fixture is bad, fail the build
# early instead of poisoning every downstream task run.
RUN python3 - <<'PY'
import datetime as dt
import os
import shutil

import lancedb
import numpy as np
import pyarrow as pa

DB = "/tmp/_pq_sanity_db"
if os.path.isdir(DB):
    shutil.rmtree(DB)

data = np.load("/app/fixtures/data.npy")
queries = np.load("/app/fixtures/queries.npy")
n_rows, dim = data.shape
ids = np.arange(n_rows, dtype=np.int64)

# Brute-force top-10 ground truth (L2).
sq_dists = ((data[None, :, :] - queries[:, None, :]) ** 2).sum(axis=-1)
gt = np.argsort(sq_dists, axis=1)[:, :10]

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), dim)),
])

db = lancedb.connect(DB)
rows = [{"id": int(ids[i]), "vector": data[i].tolist()} for i in range(n_rows)]

recalls = {}
for m in (4, 8, 16):
    name = f"sanity_m{m}"
    if name in db.table_names():
        db.drop_table(name)
    tbl = db.create_table(name, data=rows, schema=schema)
    tbl.create_index(
        vector_column_name="vector",
        index_type="IVF_PQ",
        metric="L2",
        num_partitions=16,
        num_sub_vectors=m,
        replace=True,
    )
    tbl.wait_for_index(["vector_idx"], timeout=dt.timedelta(minutes=2))

    hits = 0
    for qi, qv in enumerate(queries):
        res = tbl.search(qv).limit(10).nprobes(16).to_list()
        got = {int(r["id"]) for r in res}
        truth = set(int(x) for x in gt[qi])
        hits += len(got & truth)
    recalls[m] = hits / (len(queries) * 10.0)

print("sanity recalls:", recalls)
assert recalls[4] <= recalls[8] + 1e-9, recalls
assert recalls[8] <= recalls[16] + 1e-9, recalls
assert recalls[16] > recalls[4], recalls
shutil.rmtree(DB)
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB IVF_PQ recall-quality sweep task: 1024x64 deterministic dataset, 30 queries, ground-truth top-10."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: candidate solution directory
#   - /app/fixtures/data.npy: 1024 x 64 float32 dataset
#   - /app/fixtures/queries.npy: 30 x 64 float32 query vectors
#   - /app/fixtures/metadata.json: fixture metadata (rows, dim, seed)
