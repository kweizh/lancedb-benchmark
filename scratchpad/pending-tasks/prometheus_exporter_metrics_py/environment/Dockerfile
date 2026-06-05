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
        "prometheus_client>=0.20" \
        requests==2.32.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /home/user/myproject/data

# Build-time sanity: vector column is documented in instructions only.
RUN mkdir -p /opt/seed

RUN cat >/opt/seed/_seed.py <<'PYEOF'
"""Idempotent seed for the prometheus_exporter_metrics_py task.

Creates `documents_${ZEALT_RUN_ID}` under /home/user/myproject/data/db
with 200 rows ({id int64, content string, vector fixed_size_list<float32, 32>})
and a native (non-Tantivy) Lance FTS index on `content`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa


def main() -> int:
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("ZEALT_RUN_ID env var must be set for seeding.", file=sys.stderr)
        return 2

    db_dir = "/home/user/myproject/data/db"
    marker = f"/home/user/myproject/data/.seeded_{run_id}"
    if os.path.exists(marker):
        print(f"Seed marker {marker} already exists; skipping.")
        return 0

    os.makedirs(db_dir, exist_ok=True)

    import lancedb  # late import: lancedb spawns native threads at import time

    rng = np.random.default_rng(2026)
    n_rows = 200
    vectors = rng.standard_normal((n_rows, 32)).astype("float32")

    # Content templates that include the token "lance" so FTS queries match.
    templates = [
        "lance is a columnar format for ml workloads, doc {i}",
        "this document {i} talks about vector search in lance",
        "row {i}: lance fts indexes support bm25 ranking",
        "entry {i} compares lance with parquet for analytics",
        "{i} - a note about lance hybrid search and reranking",
    ]
    contents = [templates[i % len(templates)].format(i=i) for i in range(n_rows)]

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 32)),
        ]
    )
    table_data = pa.table(
        {
            "id": pa.array(list(range(n_rows)), type=pa.int64()),
            "content": pa.array(contents, type=pa.string()),
            "vector": pa.array(
                [v.tolist() for v in vectors],
                type=pa.list_(pa.float32(), 32),
            ),
        },
        schema=schema,
    )

    db = lancedb.connect(db_dir)
    table_name = f"documents_{run_id}"
    if table_name in db.table_names():
        db.drop_table(table_name)
    tbl = db.create_table(table_name, data=table_data, mode="overwrite")

    # Native Lance FTS index (no tantivy).
    tbl.create_fts_index("content", use_tantivy=False)

    assert tbl.count_rows() == n_rows, f"seed: expected {n_rows} rows, got {tbl.count_rows()}"

    with open(marker, "w") as f:
        f.write("ok\n")
    print(f"Seeded {table_name} with {n_rows} rows + native FTS index.")
    # lancedb 0.25.3 can SIGABRT during interpreter teardown; exit hard.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

RUN cat >/usr/local/bin/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u
if [ -z "${ZEALT_RUN_ID:-}" ]; then
  echo "ZEALT_RUN_ID env var must be set." >&2
  exit 2
fi
python3 /opt/seed/_seed.py
status=$?
if [ "$status" -ne 0 ]; then
  echo "Seed script failed with status $status" >&2
  exit "$status"
fi
trap : TERM INT
sleep infinity & wait
SHEOF
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + prometheus_client task: solver builds a metrics-instrumented Search wrapper exposing /metrics on 127.0.0.1:9100."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, prometheus_client>=0.20, requests
#   - /home/user/myproject: solver workspace
#   - /opt/seed/_seed.py: builds the LanceDB documents_${ZEALT_RUN_ID} table at container start
#   - /usr/local/bin/entrypoint.sh: idempotent seed then `sleep infinity`
