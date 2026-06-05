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
RUN mkdir -p /home/user/myproject /home/user/myproject/lancedb_data /app

# Seed script: builds base table with IVF_PQ index, writes pointer file. Idempotent.
RUN cat >/app/_seed.py <<'PYEOF'
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pyarrow as pa
import lancedb

DB_PATH = "/home/user/myproject/lancedb_data"
POINTER_PATH = "/app/index_pointer.json"

run_id = os.environ.get("ZEALT_RUN_ID")
if not run_id:
    print("ZEALT_RUN_ID not set; aborting seed.", file=sys.stderr)
    sys.exit(1)

base_name = f"vectors_{run_id}"
marker = f"/app/.seeded_{run_id}"

# Initialize / overwrite pointer file regardless of marker, so verifier reruns find a clean state.
with open(POINTER_PATH, "w") as f:
    json.dump({"active": base_name}, f)

if os.path.exists(marker):
    print(f"Already seeded for run_id={run_id}; pointer reset.")
    os._exit(0)

db = lancedb.connect(DB_PATH)

# Build 320 rows of 64-d float32 vectors.
rng = np.random.default_rng(2026)
n_rows = 320
dim = 64
vectors = rng.standard_normal((n_rows, dim)).astype(np.float32)
ids = np.arange(n_rows, dtype=np.int64)
payload = [f"item-{i}" for i in range(n_rows)]

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), dim)),
])

tbl_data = pa.Table.from_pydict(
    {"id": ids, "payload": payload, "vector": vectors.tolist()},
    schema=schema,
)

if base_name in db.table_names():
    db.drop_table(base_name)

tbl = db.create_table(base_name, data=tbl_data)

tbl.create_index(
    metric="l2",
    vector_column_name="vector",
    index_type="IVF_PQ",
    num_partitions=8,
    num_sub_vectors=8,
    replace=True,
)
tbl.wait_for_index(["vector_idx"], timeout=timedelta(seconds=120))

with open(marker, "w") as f:
    f.write("ok\n")

print(f"Seeded {base_name} with {n_rows} rows and IVF_PQ index.")
os._exit(0)
PYEOF

# Entrypoint: export BASE_TABLE_NAME, run seed, then sleep forever.
RUN cat >/app/entrypoint.sh <<'BASHEOF'
#!/bin/bash
set -u
if [ -z "${ZEALT_RUN_ID:-}" ]; then
    echo "ZEALT_RUN_ID must be set in the environment." >&2
    exit 1
fi
export BASE_TABLE_NAME="vectors_${ZEALT_RUN_ID}"
python3 /app/_seed.py
trap : TERM INT
sleep infinity & wait
BASHEOF
RUN chmod +x /app/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB zero-downtime vector index swap task with a pre-seeded IVF_PQ base table and a JSON alias pointer."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: solver project directory
#   - /home/user/myproject/lancedb_data: LanceDB connection root
#   - /app/_seed.py: seeds base table vectors_${ZEALT_RUN_ID} (320 rows, 64-d, IVF_PQ index)
#   - /app/index_pointer.json: alias pointer rewritten by /app/entrypoint.sh on every container start
#   - /app/entrypoint.sh: runs seed and waits forever
