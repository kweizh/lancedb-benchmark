FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
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
RUN mkdir -p /app/lancedb

ENV LANCEDB_URI=/app/lancedb

# Seed script: creates a deterministic benchmark table named vectors_${ZEALT_RUN_ID}
# with 5000 rows of 64-dim float32 vectors. Runs at container start (the table name
# depends on the runtime ZEALT_RUN_ID). Idempotent via a per-run marker file.
RUN cat >/app/_seed.py <<'PYEOF'
import os
import sys

import numpy as np
import pyarrow as pa
import lancedb

URI = os.environ.get("LANCEDB_URI", "/app/lancedb")
RUN_ID = os.environ.get("ZEALT_RUN_ID", "")
if not RUN_ID:
    print("ZEALT_RUN_ID is not set; skipping seed", file=sys.stderr)
    sys.exit(1)

N = 5000
DIM = 64
SEED = 2026
TABLE = f"vectors_{RUN_ID}"
MARKER = f"/app/.seeded_{RUN_ID}"

if os.path.exists(MARKER):
    print(f"{TABLE} already seeded")
    sys.stdout.flush()
    os._exit(0)

os.makedirs(URI, exist_ok=True)

rng = np.random.default_rng(SEED)
vecs = rng.standard_normal((N, DIM)).astype("float32")
ids = np.arange(N, dtype=np.int64)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])
data = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "vector": pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), DIM)),
    },
    schema=schema,
)

db = lancedb.connect(URI)
db.create_table(TABLE, data=data, mode="overwrite")

with open(MARKER, "w") as f:
    f.write("ok")

print(f"seeded {TABLE} with {N} rows ({DIM}-dim)")
sys.stdout.flush()
# lancedb 0.25.3 can raise SIGABRT during interpreter teardown; exit hard.
os._exit(0)
PYEOF

RUN cat >/app/entrypoint.sh <<'SHEOF'
#!/bin/bash
python3 /app/_seed.py || true
exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
SHEOF
RUN chmod +x /app/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB IVF_PQ recall/latency parameter sweep task; seeds a deterministic 5000-row 64-dim vector table at container start."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: lancedb==0.25.3 (pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3)
#   - /home/user/myproject: candidate project directory (solution.py goes here)
#   - /app/lancedb: embedded LanceDB database directory (LANCEDB_URI)
#   - /app/_seed.py + /app/entrypoint.sh: seed vectors_${ZEALT_RUN_ID} (5000 x 64-dim) at start

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
