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
RUN mkdir -p /home/user/myproject /home/user/myproject/data

# ------------------------------------------------------------
# Seed script: writes a deterministic ACL corpus to LanceDB at
# /home/user/myproject/data, table name = documents_${ZEALT_RUN_ID}.
# Triggered by /usr/local/bin/entrypoint.sh on first container start.
# ------------------------------------------------------------
RUN cat >/usr/local/bin/_seed.py <<'PYEOF'
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

DB_DIR = "/home/user/myproject/data"
RUN_ID = os.environ["ZEALT_RUN_ID"]
TABLE = f"documents_{RUN_ID}"

NUM_ROWS = 300
DIM = 32
NUM_OWNERS = 10
NUM_TEAMS = 8
VISIBILITIES = ["public", "team", "private"]

rng = np.random.default_rng(2026)
# Draw vectors first to anchor the RNG stream against verifier expectations.
vectors = rng.standard_normal((NUM_ROWS, DIM)).astype(np.float32)

# Cyclic visibility assignment so each class has exactly NUM_ROWS / 3 entries.
visibility = np.array([VISIBILITIES[i % 3] for i in range(NUM_ROWS)], dtype=object)

owner_idx = rng.integers(0, NUM_OWNERS, size=NUM_ROWS)
team_idx = rng.integers(0, NUM_TEAMS, size=NUM_ROWS)

owner_id = np.array([f"user_{i:02d}" for i in owner_idx], dtype=object)
team_id = np.array([f"team_{i}" for i in team_idx], dtype=object)

# Force at least one row owned by each canonical user_id for branch coverage.
# (The verifier exercises U1..U5 -> user_00..user_04.)
for canonical_user in range(5):
    target_rows = np.flatnonzero((visibility == "private") & (owner_id == f"user_{canonical_user:02d}"))
    if target_rows.size == 0:
        # Steal one private-row slot and reassign to this user.
        private_rows = np.flatnonzero(visibility == "private")
        owner_id[private_rows[canonical_user]] = f"user_{canonical_user:02d}"

# Force at least one team-row per team_0..team_7 for branch coverage.
for t in range(NUM_TEAMS):
    target_rows = np.flatnonzero((visibility == "team") & (team_id == f"team_{t}"))
    if target_rows.size == 0:
        team_rows = np.flatnonzero(visibility == "team")
        team_id[team_rows[t]] = f"team_{t}"

ids = np.arange(NUM_ROWS, dtype=np.int64)
texts = np.array([f"Document {i}" for i in range(NUM_ROWS)], dtype=object)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.utf8()),
    pa.field("owner_id", pa.utf8()),
    pa.field("visibility", pa.utf8()),
    pa.field("team_id", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

table_data = pa.table({
    "id": ids,
    "text": pa.array(texts, type=pa.utf8()),
    "owner_id": pa.array(owner_id, type=pa.utf8()),
    "visibility": pa.array(visibility, type=pa.utf8()),
    "team_id": pa.array(team_id, type=pa.utf8()),
    "vector": pa.array(list(vectors.tolist()), type=pa.list_(pa.float32(), DIM)),
}, schema=schema)

os.makedirs(DB_DIR, exist_ok=True)
db = lancedb.connect(DB_DIR)
if TABLE in db.table_names():
    db.drop_table(TABLE)
db.create_table(TABLE, data=table_data)

# Sanity: all three visibility classes + each canonical user must have rows.
unique_vis = set(visibility.tolist())
assert unique_vis == set(VISIBILITIES), f"visibility classes missing: {unique_vis}"
for u in range(5):
    n = int(((visibility == "private") & (owner_id == f"user_{u:02d}")).sum())
    assert n >= 1, f"no private rows for user_{u:02d}"
for t in range(NUM_TEAMS):
    n = int(((visibility == "team") & (team_id == f"team_{t}")).sum())
    assert n >= 1, f"no team rows for team_{t}"

print(f"[seed] wrote {NUM_ROWS} rows to {TABLE} under {DB_DIR}")
sys.stdout.flush()
os._exit(0)
PYEOF

RUN cat >/usr/local/bin/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u

: "${ZEALT_RUN_ID:?ZEALT_RUN_ID must be set}"

MARKER="/home/user/myproject/data/.seeded_${ZEALT_RUN_ID}"

if [ ! -f "${MARKER}" ]; then
    echo "[entrypoint] seeding LanceDB for run ${ZEALT_RUN_ID}..."
    python3 /usr/local/bin/_seed.py
    touch "${MARKER}"
    echo "[entrypoint] seed complete."
else
    echo "[entrypoint] LanceDB already seeded for run ${ZEALT_RUN_ID}; skipping."
fi

exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
SHEOF

RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/_seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB row-level ACL filtering task: 300-row documents table seeded at entrypoint, candidate implements ACLSearch with server-side SQL filter."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: candidate project root
#   - /home/user/myproject/data: LanceDB connection directory seeded at entrypoint
#   - /usr/local/bin/_seed.py: deterministic 300-row ACL corpus generator (numpy seed=2026)
#   - /usr/local/bin/entrypoint.sh: idempotent seed (marker file) + sleep infinity
