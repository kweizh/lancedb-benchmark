FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        pytest-xprocess==1.0.2 \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        websockets==13.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Provide embed.py helper at the project root. Used both by the entrypoint
# seed step and by the candidate's server for vector/hybrid query embedding.
RUN mkdir -p /home/user/myproject && \
    cat > /home/user/myproject/embed.py <<'PYEOF'
"""Deterministic 32-d unit embedding helper used by both the entrypoint seed
script and the candidate's WebSocket server. The output for a given input
string is identical across processes because it is derived from a SHA256
seed fed into numpy.random.default_rng.
"""
from __future__ import annotations

import hashlib

import numpy as np


def embed_text(text: str) -> np.ndarray:
    """Return a deterministic 32-d unit float32 vector for ``text``."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(32).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec = np.zeros(32, dtype=np.float32)
        vec[0] = 1.0
        return vec
    return (vec / norm).astype(np.float32)
PYEOF

# Seed script and entrypoint live under /app. The entrypoint runs the seed
# once per (ZEALT_RUN_ID, LANCEDB_PATH) combination and then sleeps to keep
# the container alive for the candidate / verifier.
RUN mkdir -p /app && \
    cat > /app/_seed.py <<'PYEOF'
"""Seed the WebSocket-search demo table with 200 rows of rigged content.

Idempotent: skips if the table already exists with the expected row count.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa


def _load_embed():
    spec = importlib.util.spec_from_file_location(
        "embed_helper", "/home/user/myproject/embed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.embed_text


def main() -> int:
    run_id = os.environ.get("ZEALT_RUN_ID")
    db_path = os.environ.get("LANCEDB_PATH", "/app/lancedb_data")
    if not run_id:
        print("ZEALT_RUN_ID is not set; refusing to seed.", file=sys.stderr)
        return 2

    os.makedirs(db_path, exist_ok=True)
    embed_text = _load_embed()

    db = lancedb.connect(db_path)
    table_name = f"docs_{run_id}"

    if table_name in db.table_names():
        tbl = db.open_table(table_name)
        if tbl.count_rows() == 200:
            print(
                f"Seed: table {table_name} already has 200 rows; skipping."
            )
            os._exit(0)
        # Stale partial table — drop and re-seed.
        db.drop_table(table_name)

    rng = np.random.default_rng(2026)
    nouns = [
        "river", "chess", "thunder", "garden", "lantern", "harbor",
        "compass", "meadow", "violin", "trellis", "pendant", "courtyard",
    ]
    verbs = [
        "drifts past", "echoes through", "rests beside", "weaves around",
        "outshines", "follows", "ignites", "balances on",
    ]
    objects = [
        "the silver moon", "an old radio", "the sleeping fox",
        "twelve lanterns", "the marble fountain", "a distant bell",
        "the autumn breeze", "the chess clock",
    ]

    rows = []
    for i in range(200):
        if i == 42:
            text = (
                "QUANTUMTOKEN42UNIQUE QUANTUMTOKEN42UNIQUE QUANTUMTOKEN42UNIQUE "
                "doc about chess and rivers and an old radio at the harbor."
            )
            vec = embed_text("QUANTUMTOKEN42UNIQUE")
        elif i in (10, 20, 30, 40):
            base = (
                f"{nouns[i % len(nouns)]} {verbs[i % len(verbs)]} "
                f"{objects[i % len(objects)]} in a quiet field"
            )
            text = f"{base} QUANTUMTOKEN42UNIQUE"
            vec = embed_text(f"row_seed_{i}")
        elif i == 99:
            text = (
                "VORTEXTOKEN99UNIQUE VORTEXTOKEN99UNIQUE VORTEXTOKEN99UNIQUE "
                "doc about clouds and bells and a marble fountain on the hill."
            )
            vec = embed_text("row_seed_99")
        elif i in (11, 22):
            base = (
                f"{nouns[(i + 3) % len(nouns)]} {verbs[(i + 1) % len(verbs)]} "
                f"{objects[(i + 5) % len(objects)]} along the cliff"
            )
            text = f"{base} VORTEXTOKEN99UNIQUE"
            vec = embed_text(f"row_seed_{i}")
        else:
            text = (
                f"{nouns[i % len(nouns)]} {verbs[i % len(verbs)]} "
                f"{objects[i % len(objects)]} on the bridge of row {i}"
            )
            vec = embed_text(f"row_seed_{i}")
        rows.append({"id": int(i), "text": text, "vector": vec.tolist()})

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 32)),
        ]
    )
    tbl = db.create_table(table_name, data=rows, schema=schema, mode="overwrite")

    # Native Lance FTS index on text column (no Tantivy).
    tbl.create_fts_index("text", use_tantivy=False, replace=True)

    # Sanity-check anchor invariants. Fail fast on a bad seed.
    n_q = tbl.count_rows(filter="text LIKE '%QUANTUMTOKEN42UNIQUE%'")
    n_v = tbl.count_rows(filter="text LIKE '%VORTEXTOKEN99UNIQUE%'")
    assert n_q >= 5, f"expected at least 5 rows with QUANTUMTOKEN42UNIQUE, got {n_q}"
    assert n_v >= 3, f"expected at least 3 rows with VORTEXTOKEN99UNIQUE, got {n_v}"

    # Verify FTS anchor returns id 42 at rank 1.
    hit = tbl.search("QUANTUMTOKEN42UNIQUE", query_type="fts").limit(1).to_list()
    assert hit and hit[0]["id"] == 42, (
        f"FTS rank-1 for QUANTUMTOKEN42UNIQUE expected id=42, got {hit}"
    )

    # Verify vector anchor returns id 42 at rank 1.
    qvec = embed_text("QUANTUMTOKEN42UNIQUE")
    vhit = tbl.search(qvec).limit(1).to_list()
    assert vhit and vhit[0]["id"] == 42, (
        f"Vector rank-1 for embed_text('QUANTUMTOKEN42UNIQUE') expected id=42, got {vhit}"
    )

    print(
        f"Seed: created table {table_name} with 200 rows + native FTS index on text."
    )
    # lancedb 0.25.3 has a teardown SIGABRT path on some runtimes; exit cleanly.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
PYEOF

RUN cat > /app/entrypoint.sh <<'SHEOF'
#!/bin/bash
set -u

: "${LANCEDB_PATH:=/app/lancedb_data}"
export LANCEDB_PATH

mkdir -p "$LANCEDB_PATH"

if [ -n "${ZEALT_RUN_ID:-}" ]; then
    MARKER="/app/.seeded_${ZEALT_RUN_ID}"
    if [ ! -f "$MARKER" ]; then
        echo "entrypoint: seeding table for ZEALT_RUN_ID=${ZEALT_RUN_ID}"
        python3 /app/_seed.py
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "entrypoint: seed failed with exit code $rc" >&2
            exit "$rc"
        fi
        touch "$MARKER"
    else
        echo "entrypoint: seed marker present ($MARKER); skipping."
    fi
else
    echo "entrypoint: ZEALT_RUN_ID not set; skipping seed." >&2
fi

exec /bin/bash -c 'trap : TERM INT; sleep infinity & wait'
SHEOF
RUN chmod +x /app/entrypoint.sh

ENV LANCEDB_PATH=/app/lancedb_data

LABEL Author="generate-harbor-tasks"
LABEL Description="WebSocket real-time search WebSocket server task on LanceDB 0.25.3 with native FTS + RRF hybrid."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf pytest-xprocess lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 websockets==13.1
#   - /home/user/myproject/embed.py: deterministic 32-d embedding helper used by seed + candidate server
#   - /app/_seed.py: seeds docs_${ZEALT_RUN_ID} with 200 rows (rigged ids 42, 99) and native FTS index
#   - /app/entrypoint.sh: runs seed idempotently then sleeps infinity
#   - LANCEDB_PATH=/app/lancedb_data
