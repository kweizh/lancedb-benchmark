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
        opentelemetry-api==1.27.0 \
        opentelemetry-sdk==1.27.0 \
        opentelemetry-exporter-otlp-proto-http==1.27.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && \
    mkdir -p /app/lancedb_data

# Seed script: creates the 50-row 32-d table named tracing_docs_${ZEALT_RUN_ID}
# with a `content` text column and a native FTS index. Runs at container start.
RUN cat >/app/_seed.py <<'PYEOF'
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

LANCEDB_DIR = "/app/lancedb_data"
DIM = 32
N_ROWS = 50

WORDS_BANK = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "quantum", "relativity", "matrix", "vector", "tensor", "graph", "search",
    "index", "neural", "embedding", "lance", "table", "metric", "cosine",
    "model", "data", "scalar", "kernel", "training", "feature", "channel",
    "signal", "filter", "fragment", "compact", "registry",
]


def main():
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("ZEALT_RUN_ID not set; aborting seed.", file=sys.stderr)
        sys.exit(1)

    table_name = f"tracing_docs_{run_id}"
    marker = f"/app/.seeded_{run_id}"
    if os.path.exists(marker):
        print(f"Already seeded for run {run_id}; skipping.")
        os._exit(0)

    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((N_ROWS, DIM)).astype(np.float32)
    contents = []
    for i in range(N_ROWS):
        chosen = rng.choice(WORDS_BANK, size=6, replace=False)
        # always seed at least the word "alpha" into a handful of rows so FTS finds matches
        if i % 5 == 0:
            chosen = np.append(chosen[:5], "alpha")
        contents.append(" ".join(map(str, chosen)))

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ])
    rows = [
        {"id": i, "content": contents[i], "vector": vectors[i].tolist()}
        for i in range(N_ROWS)
    ]
    tbl_data = pa.Table.from_pylist(rows, schema=schema)

    db = lancedb.connect(LANCEDB_DIR)
    if table_name in db.table_names():
        db.drop_table(table_name)
    tbl = db.create_table(table_name, data=tbl_data)
    # Build a native (non-Tantivy) FTS index so FTS queries are answerable.
    tbl.create_fts_index("content", use_tantivy=False, replace=True)

    with open(marker, "w") as f:
        f.write("ok")
    print(f"Seeded {table_name} with {tbl.count_rows()} rows.")
    # lancedb 0.25.3 can SIGABRT during interpreter teardown; exit fast.
    os._exit(0)


main()
PYEOF

RUN cat >/app/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u

if [ -z "${ZEALT_RUN_ID:-}" ]; then
    echo "[entrypoint] ZEALT_RUN_ID is not set; refusing to start." >&2
    exit 1
fi

if [ ! -f "/app/.seeded_${ZEALT_RUN_ID}" ]; then
    echo "[entrypoint] Seeding LanceDB table for run ${ZEALT_RUN_ID}..."
    python3 /app/_seed.py || { echo "[entrypoint] seed failed" >&2; exit 1; }
fi

# Keep the container alive for the evaluation lifecycle.
trap : TERM INT
sleep infinity &
wait
SHEOF

RUN chmod +x /app/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + OpenTelemetry tracing task: 50-row 32-d table seeded at entrypoint with native FTS on a content column; candidate must instrument connect/search/materialize with OTel spans exported as JSONL to /tmp/otel_spans.jsonl."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 (verified pin set)
#   - opentelemetry-api/sdk/exporter-otlp-proto-http 1.27.0
#   - /app/_seed.py + /app/entrypoint.sh: seed the 50-row 32-d table named
#     tracing_docs_${ZEALT_RUN_ID} into /app/lancedb_data with a native FTS index on
#     the `content` column, then keep the container alive (sleep infinity).
#   - /home/user/myproject: empty project directory where the executor builds
#     solution.py + run_queries.py.
