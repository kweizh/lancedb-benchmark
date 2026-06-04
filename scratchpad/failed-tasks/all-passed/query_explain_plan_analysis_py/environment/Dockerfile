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
RUN mkdir -p /home/user/myproject /home/user/lancedb_data

# Seed the LanceDB database with deterministic data, build the FTS index, and
# save the query vector that the candidate solution must consume.
RUN cat >/tmp/seed_data.py <<'PYEOF'
import os
import sys
import numpy as np
import pyarrow as pa
import lancedb

URI = "/home/user/lancedb_data"
TABLE = "docs"
N_ROWS = 320
NDIM = 16
SEED = 2026

rng = np.random.default_rng(SEED)
vectors = rng.standard_normal((N_ROWS, NDIM)).astype(np.float32)

categories = ["alpha", "beta", "gamma", "delta", "epsilon"]
keywords = [
    "lancedb is a vector database for ai applications",
    "the quick brown fox jumps over the lazy dog",
    "modern search engines combine fts and vector similarity",
    "structured embeddings power retrieval augmented generation",
    "an index speeds up nearest neighbor lookups significantly",
]

rows = []
for i in range(N_ROWS):
    cat = categories[i % len(categories)]
    base = keywords[i % len(keywords)]
    rows.append({
        "id": i,
        "vector": vectors[i].tolist(),
        "text": f"document {i} category {cat}: {base}",
        "category": cat,
    })

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), NDIM)),
    pa.field("text", pa.string()),
    pa.field("category", pa.string()),
])

db = lancedb.connect(URI)
tbl = db.create_table(TABLE, data=rows, schema=schema, mode="overwrite")
tbl.create_fts_index("text", use_tantivy=False, replace=True)

assert tbl.count_rows() == N_ROWS, f"expected {N_ROWS} rows, got {tbl.count_rows()}"
indices = tbl.list_indices()
fts_cols = [list(getattr(i, "columns", [])) for i in indices]
assert any(cols == ["text"] for cols in fts_cols), f"FTS index not found: {indices}"

# Save the deterministic query vector that the candidate must read.
qv = rng.standard_normal(NDIM).astype(np.float32)
np.save("/home/user/myproject/query_vector.npy", qv)

print("seeded rows:", tbl.count_rows(), "indices:", indices)

# Workaround for the lancedb 0.25.3 SIGABRT at interpreter shutdown documented
# in memory.
os._exit(0)
PYEOF
RUN python3 /tmp/seed_data.py && rm /tmp/seed_data.py

ENV LANCEDB_URI=/home/user/lancedb_data
ENV TABLE_NAME=docs
ENV CATEGORY_FILTER=alpha
ENV FTS_QUERY=lancedb

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 task environment with a pre-seeded 320-row docs table, native FTS index on text, and a deterministic query vector for explain_plan/analyze_plan introspection."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 for the task and the verifier
#   - /home/user/myproject: project directory where the candidate writes solution.py and report.json
#   - /home/user/myproject/query_vector.npy: deterministic 16-d float32 query vector (seed=2026)
#   - /home/user/lancedb_data: LanceDB database directory containing the pre-seeded docs table with FTS index on text
