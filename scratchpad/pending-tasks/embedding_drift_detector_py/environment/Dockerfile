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
        scikit-learn==1.5.2 \
        "scipy>=1.13,<2" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Seed script: deterministic fixture for the baseline (N(0, I)) and current
# (shifted N(mu, I) with non-zero mu in 3 specific dims) 64-d float32 tables.
# Tables are suffixed with ZEALT_RUN_ID so concurrent trials never collide.
# os._exit(0) avoids the lancedb 0.25.3 teardown SIGABRT.
RUN cat >/home/user/myproject/_seed.py <<'PY'
import os
import sys

import numpy as np
import pyarrow as pa
import lancedb

RUN_ID = os.environ.get("ZEALT_RUN_ID")
if not RUN_ID:
    print("ZEALT_RUN_ID not set; refusing to seed.", file=sys.stderr)
    sys.exit(2)

DATA_DIR = "/home/user/myproject/data"
os.makedirs(DATA_DIR, exist_ok=True)

N = 1000
DIM = 64
SHIFT_DIMS = [3, 27, 51]
SHIFT_MAG = 2.5

rng_b = np.random.default_rng(20260101)
baseline_vecs = rng_b.standard_normal(size=(N, DIM)).astype(np.float32)

rng_c = np.random.default_rng(20260102)
current_vecs = rng_c.standard_normal(size=(N, DIM)).astype(np.float32)
for d in SHIFT_DIMS:
    current_vecs[:, d] += SHIFT_MAG

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

def _to_table(vecs):
    ids = pa.array(np.arange(vecs.shape[0], dtype=np.int64))
    vecs_list = pa.array(vecs.tolist(), type=pa.list_(pa.float32(), DIM))
    return pa.Table.from_arrays([ids, vecs_list], schema=schema)

db = lancedb.connect(DATA_DIR)
baseline_name = f"baseline_{RUN_ID}"
current_name = f"current_{RUN_ID}"

existing = set(db.table_names())
if baseline_name in existing:
    db.drop_table(baseline_name)
if current_name in existing:
    db.drop_table(current_name)

db.create_table(baseline_name, data=_to_table(baseline_vecs))
db.create_table(current_name, data=_to_table(current_vecs))

print(f"Seeded {baseline_name} and {current_name} ({N} rows, {DIM}-d) under {DATA_DIR}")
os._exit(0)
PY

RUN cat >/usr/local/bin/entrypoint.sh <<'SH'
#!/bin/bash
set -u
RUN_ID="${ZEALT_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
    echo "[entrypoint] ZEALT_RUN_ID is empty; skipping seed." >&2
else
    MARKER="/home/user/myproject/data/.seeded_${RUN_ID}"
    if [ ! -f "$MARKER" ]; then
        echo "[entrypoint] Seeding LanceDB tables for run ${RUN_ID}..."
        cd /home/user/myproject && python3 _seed.py
        mkdir -p /home/user/myproject/data
        touch "$MARKER"
    else
        echo "[entrypoint] Marker $MARKER present; skipping reseed."
    fi
fi
exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
SH
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="Embedding drift detector: baseline N(0,I) vs shifted N(mu,I) 64-d LanceDB tables for k-means + KL/JS analysis."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4,
#                    pandas==2.2.3, scikit-learn==1.5.2, scipy>=1.13
#   - /home/user/myproject: project root for the candidate solution
#   - /home/user/myproject/_seed.py: fixture script writing 1000-row 64-d baseline / current tables
#   - /usr/local/bin/entrypoint.sh: seeds tables once per run via .seeded_${ZEALT_RUN_ID} marker
