FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        redis-server && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        'redis>=5.0,<6' && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && mkdir -p /home/user/myproject/lancedb_data

# Seed script: creates the docs_${ZEALT_RUN_ID} table with 200 rows and a native FTS index
RUN cat >/opt/_seed.py <<'EOF'
import os
import sys
import numpy as np
import pyarrow as pa
import lancedb

run_id = os.environ.get("ZEALT_RUN_ID")
if not run_id:
    print("ZEALT_RUN_ID is not set; skipping seed.", file=sys.stderr)
    sys.exit(0)

db_path = "/home/user/myproject/lancedb_data"
os.makedirs(db_path, exist_ok=True)

db = lancedb.connect(db_path)
table_name = f"docs_{run_id}"

if table_name in db.table_names():
    print(f"Table {table_name} already exists; skipping seed.")
    os._exit(0)

rng = np.random.default_rng(2026)
n_rows = 200
dim = 32

ids = list(range(n_rows))
contents = [
    f"document number {i} about topic {i % 7} with keyword sample"
    for i in range(n_rows)
]
vectors = rng.standard_normal((n_rows, dim)).astype(np.float32)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("content", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), dim)),
])

batch = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "content": pa.array(contents, type=pa.string()),
        "vector": pa.array(vectors.tolist(), type=pa.list_(pa.float32(), dim)),
    },
    schema=schema,
)

tbl = db.create_table(table_name, batch, mode="create")
# Native FTS index so the candidate's text query path can use FTS.
tbl.create_fts_index("content", use_tantivy=False)

print(f"Seeded {table_name} with {tbl.count_rows()} rows.")
# os._exit to avoid lancedb 0.25.3 SIGABRT on interpreter teardown.
os._exit(0)
EOF

# Entrypoint: idempotently start redis-server, seed the table, then hand off.
RUN cat >/opt/entrypoint.sh <<'EOF'
#!/bin/bash
set -u

# Start Redis daemon if not already running on 127.0.0.1:6379
if ! redis-cli -h 127.0.0.1 -p 6379 PING >/dev/null 2>&1; then
    redis-server --daemonize yes --bind 127.0.0.1 --port 6379 --save "" --appendonly no
    # Wait until it's reachable
    for _ in $(seq 1 20); do
        if redis-cli -h 127.0.0.1 -p 6379 PING >/dev/null 2>&1; then
            break
        fi
        sleep 0.25
    done
fi

# Seed LanceDB once per ZEALT_RUN_ID.
RUN_ID="${ZEALT_RUN_ID:-norunid}"
MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "${MARKER}" ]; then
    python3 /opt/_seed.py && touch "${MARKER}" || echo "Seed script failed" >&2
fi

# Hand off to the supplied command, or default to sleep.
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    trap : TERM INT
    sleep infinity & wait
fi
EOF
RUN chmod +x /opt/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + Redis cache task: redis-server daemon on localhost:6379 plus a pre-seeded LanceDB table docs_<run_id> for cache-layer evaluation."

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep redis-server
#   - pip3 packages: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 redis>=5.0
#   - /opt/_seed.py: seeds docs_${ZEALT_RUN_ID} with 200 rows + native FTS index
#   - /opt/entrypoint.sh: idempotent: starts redis-server daemon and runs _seed.py once per run-id
#   - /home/user/myproject: project directory for candidate solution.py
#   - /home/user/myproject/lancedb_data: LanceDB data directory
