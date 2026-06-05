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
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject

# The events table is seeded at container start via the entrypoint script
# because the table name depends on ZEALT_RUN_ID which is only available at
# runtime. The seed is idempotent via a marker file inside the project dir.
RUN cat >/home/user/myproject/_seed.py <<'PYEOF'
"""Seed the LanceDB `events_${ZEALT_RUN_ID}` table with 1,000 deterministic
rows: 32-d float32 vectors, int64 unix-second timestamps spanning
2024-01-01T00:00:00Z .. 2026-01-01T00:00:00Z, 8 distinct event types, and
short string payloads."""
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

run_id = os.environ.get("ZEALT_RUN_ID", "")
if not run_id:
    print("[seed] ERROR: ZEALT_RUN_ID is required.", file=sys.stderr)
    sys.exit(1)

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
TABLE_NAME = f"events_{run_id}"
MARKER = os.path.join(PROJECT_DIR, f".seeded_{run_id}")

if os.path.exists(MARKER):
    print(f"[seed] Already seeded ({MARKER}), skipping.", flush=True)
    os._exit(0)

os.makedirs(DB_DIR, exist_ok=True)

N_ROWS = 1000
DIM = 32
SEED = 2026

# Unix-second range: [2024-01-01T00:00:00Z, 2026-01-01T00:00:00Z)
TS_LOW = 1704067200   # 2024-01-01
TS_HIGH = 1767225600  # 2026-01-01

EVENT_TYPES = [
    "click", "view", "purchase", "login",
    "logout", "search", "scroll", "share",
]

rng = np.random.default_rng(SEED)

ids = np.arange(N_ROWS, dtype=np.int64)
timestamps = rng.integers(TS_LOW, TS_HIGH, size=N_ROWS, dtype=np.int64)
event_type_idx = rng.integers(0, len(EVENT_TYPES), size=N_ROWS)
event_types = [EVENT_TYPES[i] for i in event_type_idx]
payloads = [f"payload-{i:04d}" for i in range(N_ROWS)]
vectors = rng.standard_normal((N_ROWS, DIM)).astype(np.float32)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("timestamp", pa.int64()),
    pa.field("event_type", pa.string()),
    pa.field("payload", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

table_data = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "timestamp": pa.array(timestamps, type=pa.int64()),
        "event_type": pa.array(event_types, type=pa.string()),
        "payload": pa.array(payloads, type=pa.string()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1).tolist(), type=pa.float32()),
            DIM,
        ),
    },
    schema=schema,
)

db = lancedb.connect(DB_DIR)
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)
tbl = db.create_table(TABLE_NAME, data=table_data, mode="create")
assert tbl.count_rows() == N_ROWS, tbl.count_rows()

# Sanity check: all 8 event types appear; timestamp range respected.
df = tbl.to_pandas()
assert set(df["event_type"].unique()) == set(EVENT_TYPES), (
    f"event_type coverage incomplete: {sorted(df['event_type'].unique())}"
)
ts_min = int(df["timestamp"].min())
ts_max = int(df["timestamp"].max())
assert TS_LOW <= ts_min and ts_max < TS_HIGH, (ts_min, ts_max)
print(
    f"[seed] Created {TABLE_NAME!r} with {tbl.count_rows()} rows. "
    f"ts range [{ts_min}, {ts_max}]",
    flush=True,
)

with open(MARKER, "w") as f:
    f.write("ok\n")
print(f"[seed] Wrote marker {MARKER}.", flush=True)

# lancedb 0.25.3 occasionally SIGABRTs on interpreter teardown after a
# write-heavy session; exit hard for deterministic build/run behavior.
os._exit(0)
PYEOF

RUN cat >/home/user/myproject/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u
RUN_ID="${ZEALT_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
    echo "[entrypoint] ERROR: ZEALT_RUN_ID is not set." >&2
    exit 1
fi
MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "$MARKER" ]; then
    python3 /home/user/myproject/_seed.py || {
        echo "[entrypoint] Seed failed; exiting." >&2
        exit 1
    }
fi
trap : TERM INT
sleep infinity & wait
SHEOF

RUN chmod +x /home/user/myproject/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB temporal-window semantic search task: 1000-row events table (suffixed by ZEALT_RUN_ID) with 32-d vectors, int64 unix timestamps (2024-01-01..2026-01-01), 8 event types, seeded at container start."

CMD ["/home/user/myproject/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /home/user/myproject/_seed.py   seeds events_${ZEALT_RUN_ID} at first start
#   - /home/user/myproject/entrypoint.sh  runs the seed (idempotent via marker) then sleeps
