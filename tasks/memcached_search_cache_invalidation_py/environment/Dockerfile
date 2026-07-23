FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        memcached procps && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        pymemcache==4.0.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Seed script: builds a deterministic LanceDB table named search_docs_${ZEALT_RUN_ID}
# with 500 rows of 32-d vectors (seeded numpy RNG) and writes fixture.json.
RUN cat > /home/user/myproject/_seed.py <<'PYEOF'
import json
import os

import lancedb
import numpy as np
import pyarrow as pa

DIM = 32
N = 500
CATEGORIES = ["A", "B", "C", "D", "E"]

run_id = os.environ.get("ZEALT_RUN_ID", "local")
project_dir = "/home/user/myproject"
db_path = os.path.join(project_dir, "lancedb")
table_name = f"search_docs_{run_id}"

rng = np.random.default_rng(2026)
vectors = rng.standard_normal((N, DIM)).astype("float32")
cat_idx = rng.integers(0, len(CATEGORIES), size=N)
categories = [CATEGORIES[i] for i in cat_idx]

schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("category", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ]
)

ids = pa.array(list(range(N)), pa.int64())
cats = pa.array(categories, pa.string())
flat = pa.array(vectors.reshape(-1).tolist(), pa.float32())
vecs = pa.FixedSizeListArray.from_arrays(flat, DIM)
arrow_table = pa.Table.from_arrays([ids, cats, vecs], schema=schema)

db = lancedb.connect(db_path)
if table_name in db.table_names():
    db.drop_table(table_name)
db.create_table(table_name, data=arrow_table)

fixture = {
    "db_path": db_path,
    "table_name": table_name,
    "dim": DIM,
    "num_rows": N,
}
with open(os.path.join(project_dir, "fixture.json"), "w") as f:
    json.dump(fixture, f)

print(f"Seeded table {table_name} with {N} rows at {db_path}")

# lancedb 0.25.3 can raise SIGABRT during interpreter teardown; exit hard.
os._exit(0)
PYEOF

# Entrypoint: start local memcached, seed the LanceDB table (idempotent), then idle.
RUN cat > /home/user/myproject/entrypoint.sh <<'SHEOF'
#!/bin/bash
mkdir -p /logs/artifacts
RUN_ID="${ZEALT_RUN_ID:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(cat /logs/artifacts/run-id 2>/dev/null || echo local)"
fi
export ZEALT_RUN_ID="$RUN_ID"

if ! pgrep -x memcached >/dev/null 2>&1; then
  memcached -d -u nobody -p 11211 -l 127.0.0.1 -m 64
fi

MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "$MARKER" ]; then
  python3 /home/user/myproject/_seed.py && touch "$MARKER"
fi

trap : TERM INT
sleep infinity & wait
SHEOF

RUN chmod +x /home/user/myproject/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB read-through search cache task with a local memcached daemon and a seeded 500-row vector table."

CMD ["/bin/bash", "/home/user/myproject/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - memcached + procps: local cache backing store started at container boot
#   - pip3: lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 pymemcache==4.0.0
#   - /home/user/myproject: project dir; _seed.py + entrypoint.sh seed search_docs_${ZEALT_RUN_ID}
#   - fixture.json (written at boot): db_path, table_name, dim, num_rows for the verifier

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
