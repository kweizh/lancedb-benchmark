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
RUN mkdir -p /home/user/myproject/data && chmod -R 0755 /home/user/myproject

# Seed script: creates documents_${ZEALT_RUN_ID} on first container start.
# The table name is run-id scoped, so seeding must happen at container start (not at build).
# Place seed assets under /home/user so the candidate's bind-mount of /home/user/myproject
# does not hide them.
RUN cat >/home/user/_seed.py <<'PYEOF'
import os
import sys
import numpy as np
import pyarrow as pa
import lancedb

DATA_DIR = "/home/user/myproject/data"
RUN_ID = os.environ.get("ZEALT_RUN_ID", "")
if not RUN_ID:
    print("ZEALT_RUN_ID not set; refusing to seed.", file=sys.stderr)
    os._exit(1)

TABLE_NAME = f"documents_{RUN_ID}"

db = lancedb.connect(DATA_DIR)
if TABLE_NAME in db.table_names():
    print(f"[seed] {TABLE_NAME} already exists; skipping.")
    os._exit(0)

rng = np.random.default_rng(2026)
N = 64
ids = list(range(N))
texts = [f"doc-{i:03d}" for i in range(N)]
vecs = rng.standard_normal((N, 16)).astype(np.float32)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), 16)),
])

rows = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "text": pa.array(texts, type=pa.utf8()),
        "vector": pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), 16)),
    },
    schema=schema,
)
db.create_table(TABLE_NAME, data=rows)
print(f"[seed] created {TABLE_NAME} with {N} rows")
# Avoid lancedb 0.25.3 teardown SIGABRT under write-heavy seed scripts.
os._exit(0)
PYEOF

RUN cat >/home/user/entrypoint.sh <<'SHEOF'
#!/bin/bash
set -u
mkdir -p /home/user/myproject/data
MARKER="/home/user/.seeded_${ZEALT_RUN_ID:-unset}"
if [ ! -f "$MARKER" ]; then
    python3 /home/user/_seed.py && touch "$MARKER"
fi
exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
SHEOF
RUN chmod +x /home/user/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB rate-limited query proxy environment: seeded documents_\${ZEALT_RUN_ID} table on entrypoint."

CMD ["/home/user/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /home/user/myproject/data: empty LanceDB directory (seeded on entrypoint)
#   - /home/user/_seed.py: idempotent seed for documents_${ZEALT_RUN_ID}
#   - /home/user/entrypoint.sh: runs seed once then sleep infinity
