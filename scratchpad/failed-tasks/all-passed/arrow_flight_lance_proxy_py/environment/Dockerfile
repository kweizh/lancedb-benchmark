FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        pytest-xprocess \
        requests \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/flight_proxy

# Seed the 200-row LanceDB `documents` table with deterministic 32-d float32 vectors.
RUN python3 - <<'PYEOF'
import os
import numpy as np
import pyarrow as pa
import lancedb

DB_DIR = "/home/user/flight_proxy/lancedb"
os.makedirs(DB_DIR, exist_ok=True)

rng = np.random.default_rng(2026)
N = 200
DIM = 32

ids = [f"doc_{i:03d}" for i in range(N)]
texts = [f"Document number {i:03d} about topic {i % 17}." for i in range(N)]
vectors = rng.standard_normal((N, DIM)).astype(np.float32)

schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), DIM)),
])

table = pa.Table.from_pydict(
    {
        "id": ids,
        "text": texts,
        "embedding": pa.array(list(vectors), type=pa.list_(pa.float32(), DIM)),
    },
    schema=schema,
)

conn = lancedb.connect(DB_DIR)
conn.create_table("documents", data=table, mode="overwrite")
print("Seeded 'documents' with", N, "rows.")
os._exit(0)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB-backed Apache Arrow Flight proxy task: 200-row documents table with 32-d float32 vectors."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy + pandas for the LanceDB-backed Flight proxy
#   - pytest-xprocess + requests for the verifier to launch and probe the Flight server
#   - /home/user/flight_proxy/lancedb: pre-seeded LanceDB database with the 200-row `documents` table
