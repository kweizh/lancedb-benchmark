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
        scikit-learn==1.5.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Seed script that creates the LanceDB `events_${ZEALT_RUN_ID}` table at container
# startup. The script is idempotent — if the table already exists with 1000 rows
# it simply re-writes the ground-truth file and exits 0.
RUN cat >/usr/local/bin/seed_events_table.py <<'PYEOF'
#!/usr/bin/env python3
"""Seed the events_<run-id> LanceDB table used by the outlier-detection task.

This script must be deterministic. It builds 1000 rows of 20-d float32 vectors:
  - 950 inliers ~ N(0, 0.5)
  - 50 outliers ~ N(5, 0.5)

Ground-truth outlier ids are persisted to .ground_truth_outliers.json so the
verifier can score the candidate's predictions.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pyarrow as pa

import lancedb

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
GT_PATH = os.path.join(PROJECT_DIR, ".ground_truth_outliers.json")


def build_fixture() -> tuple[list[dict], list[int]]:
    rng = np.random.default_rng(2026)
    n_inliers = 950
    n_outliers = 50
    dim = 20

    inliers = rng.normal(loc=0.0, scale=0.5, size=(n_inliers, dim)).astype(np.float32)
    outliers = rng.normal(loc=5.0, scale=0.5, size=(n_outliers, dim)).astype(np.float32)

    # Build full row table with ids 0..999, deterministically shuffle so
    # outliers are interleaved (not clustered at the tail).
    all_vecs = np.vstack([inliers, outliers])
    labels = np.array([0] * n_inliers + [1] * n_outliers)
    perm = rng.permutation(len(all_vecs))
    all_vecs = all_vecs[perm]
    labels = labels[perm]

    rows: list[dict] = []
    outlier_ids: list[int] = []
    for idx, (vec, label) in enumerate(zip(all_vecs, labels)):
        rows.append(
            {
                "id": int(idx),
                "ts": int(1_700_000_000 + idx),
                "vector": vec.tolist(),
            }
        )
        if int(label) == 1:
            outlier_ids.append(int(idx))

    outlier_ids.sort()
    return rows, outlier_ids


def main() -> int:
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("[seed] ZEALT_RUN_ID is not set; aborting seed.", file=sys.stderr)
        return 1

    os.makedirs(LANCEDB_DIR, exist_ok=True)

    rows, outlier_ids = build_fixture()
    with open(GT_PATH, "w") as f:
        json.dump(outlier_ids, f)
    os.chmod(GT_PATH, 0o644)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("ts", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), 20)),
        ]
    )

    db = lancedb.connect(LANCEDB_DIR)
    table_name = f"events_{run_id}"
    existing = set(db.table_names())
    if table_name in existing:
        tbl = db.open_table(table_name)
        if tbl.count_rows() == 1000 and "is_outlier" not in tbl.schema.names:
            print(f"[seed] Table {table_name} already exists with 1000 rows; skipping.")
            return 0
        # Drop and recreate to guarantee a clean initial state for re-runs.
        db.drop_table(table_name)

    arrays = [
        pa.array([r["id"] for r in rows], type=pa.int64()),
        pa.array([r["ts"] for r in rows], type=pa.int64()),
        pa.array([r["vector"] for r in rows], type=pa.list_(pa.float32(), 20)),
    ]
    arrow_tbl = pa.Table.from_arrays(arrays, schema=schema)
    db.create_table(table_name, data=arrow_tbl, mode="create")

    print(
        f"[seed] Created LanceDB table {table_name} with {len(rows)} rows "
        f"and {len(outlier_ids)} ground-truth outliers."
    )
    return 0


if __name__ == "__main__":
    rc = main()
    # lancedb 0.25.3 occasionally raises during interpreter teardown — use os._exit
    # to avoid masking a successful run with a SIGABRT at finalisation.
    os._exit(rc)
PYEOF
RUN chmod 0755 /usr/local/bin/seed_events_table.py

RUN cat >/usr/local/bin/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u
# Always (re)seed the LanceDB fixture before yielding control to the harness.
python3 /usr/local/bin/seed_events_table.py || true
exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
SHEOF
RUN chmod 0755 /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB outlier-detection task: 1000-row events_<run-id> table seeded at container start; candidate fits IsolationForest and writes back is_outlier."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - pip3: lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2
#   - /usr/local/bin/seed_events_table.py: seed script (idempotent, run at container start)
#   - /usr/local/bin/entrypoint.sh: container entrypoint that seeds then sleeps
#   - /home/user/myproject: project directory where the candidate writes solution.py / run.py
#   - /home/user/myproject/lancedb_data/: LanceDB DB seeded at startup with events_<run-id>
#   - /home/user/myproject/.ground_truth_outliers.json: 50 ground-truth outlier ids for the verifier
