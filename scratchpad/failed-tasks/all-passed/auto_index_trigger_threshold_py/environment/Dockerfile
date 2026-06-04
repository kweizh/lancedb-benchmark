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
RUN mkdir -p /home/user/myproject && \
    mkdir -p /workspace/db && \
    mkdir -p /workspace/output && \
    mkdir -p /opt

# Write the idempotent seed/reset script that prepares the 100-row 'vectors' table
# with NO vector index and dumps the rigged sentinel id + query vector to
# /home/user/myproject/.expected.json.  The verifier re-runs this script before
# pytest to restore the starting baseline.
RUN cat >/opt/seed_state.py <<'PYEOF'
"""Seed and reset the LanceDB workspace to the baseline expected by the task.

Wipes /workspace/db, creates a 'vectors' table with exactly 100 deterministic
rows (no vector index), and writes /home/user/myproject/.expected.json with the
rigged sentinel id and the query vector that the verifier uses to check
correctness.
"""
import json
import os
import shutil

import numpy as np
import pyarrow as pa

import lancedb

LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
VECTORS_TABLE = os.environ.get("VECTORS_TABLE", "vectors")
EXPECTED_PATH = "/home/user/myproject/.expected.json"
DIM = 64
SEED = 2026
SEED_ROWS = 100
SENTINEL_ID = 0


def main() -> None:
    # Wipe and recreate the LanceDB directory.
    if os.path.isdir(LANCEDB_URI):
        shutil.rmtree(LANCEDB_URI)
    os.makedirs(LANCEDB_URI, exist_ok=True)
    os.makedirs(os.path.dirname(EXPECTED_PATH), exist_ok=True)

    rng = np.random.default_rng(SEED)
    base = rng.standard_normal((SEED_ROWS, DIM)).astype(np.float32)

    # Force the sentinel row at id=0 to have a unique direction so that its
    # cosine distance to itself is 0 and its cosine distance to every other
    # random/standard-normal vector is meaningfully > 0.
    sentinel_vec = np.zeros(DIM, dtype=np.float32)
    sentinel_vec[0] = 10.0
    sentinel_vec[1] = -10.0
    sentinel_vec[2] = 7.0
    base[SENTINEL_ID, :] = sentinel_vec

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
        ]
    )

    ids = list(range(SEED_ROWS))
    rows = [
        {"id": int(ids[i]), "vector": base[i].tolist()}
        for i in range(SEED_ROWS)
    ]

    db = lancedb.connect(LANCEDB_URI)
    # Drop pre-existing tables to keep the reset idempotent.
    for name in list(db.table_names()):
        db.drop_table(name)

    db.create_table(VECTORS_TABLE, data=rows, schema=schema, mode="overwrite")

    with open(EXPECTED_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "sentinel_id": int(SENTINEL_ID),
                "query_vector": sentinel_vec.tolist(),
                "dim": DIM,
                "seed_rows": SEED_ROWS,
                "seed": SEED,
            },
            fh,
        )

    print(
        f"Seeded {VECTORS_TABLE!r} at {LANCEDB_URI} with {SEED_ROWS} rows, "
        f"sentinel_id={SENTINEL_ID}, expected={EXPECTED_PATH}."
    )

    # Workaround for lancedb 0.25.3 occasionally SIGABRT-ing during interpreter
    # shutdown after a write-heavy seed run.
    os._exit(0)


if __name__ == "__main__":
    main()
PYEOF

# Bake the baseline by running the seed script once at image build time so the
# 'vectors' table and .expected.json exist before any executor runs.
RUN python3 /opt/seed_state.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB 0.25.3 environment for an IndexedTable wrapper that auto-builds IVF_PQ when the row count crosses configured thresholds and writes an audit log."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 pinned (verified to import cleanly on ubuntu:24.04)
#   - pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 pinned
#   - /home/user/myproject: executor working directory for solution.py
#   - /workspace/db: LanceDB URI pre-seeded with a 100-row 'vectors' table (no index)
#   - /opt/seed_state.py: idempotent re-seed script invoked by the verifier
#   - /home/user/myproject/.expected.json: rigged sentinel id + query vector fixture
