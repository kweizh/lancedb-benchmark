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
        "networkx>=3.0" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject/lancedb_data

# Seed script: creates embeddings_${ZEALT_RUN_ID} (500 rows × 64-d float32) on container start.
RUN cat >/usr/local/bin/_seed.py <<'PYEOF'
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

LANCEDB_DIR = "/home/user/myproject/lancedb_data"


def main() -> None:
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("[seed] ZEALT_RUN_ID is not set; nothing to seed.", file=sys.stderr)
        os._exit(0)

    marker = f"/home/user/myproject/.seeded_{run_id}"
    table_name = f"embeddings_{run_id}"

    db = lancedb.connect(LANCEDB_DIR)
    if os.path.exists(marker) and table_name in db.table_names():
        print(f"[seed] {table_name} already present; skipping.")
        os._exit(0)

    rng = np.random.default_rng(2026)
    n_rows = 500
    dim = 64
    vectors = rng.standard_normal((n_rows, dim)).astype(np.float32)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )
    ids = pa.array(list(range(n_rows)), type=pa.int64())
    vec_arr = pa.array(
        [v.tolist() for v in vectors],
        type=pa.list_(pa.float32(), dim),
    )
    table = pa.Table.from_arrays([ids, vec_arr], schema=schema)

    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, data=table)
    with open(marker, "w") as f:
        f.write("ok")
    print(f"[seed] Created {table_name} with {n_rows} rows of dim={dim}.")
    # lancedb 0.25.3 sometimes SIGABRTs during interpreter teardown; bypass it.
    os._exit(0)


main()
PYEOF

# Entrypoint: seed embeddings, then sleep forever.
RUN cat >/usr/local/bin/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -u
python3 /usr/local/bin/_seed.py || true
trap : TERM INT
sleep infinity & wait
SHEOF
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/_seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB KNN graph construction task: 500x64-d embeddings table seeded at container start; candidate builds knn_edges_<run_id> + find_path BFS."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, networkx>=3.0
#   - /home/user/myproject/lancedb_data: LanceDB store
#   - /usr/local/bin/_seed.py: seeds embeddings_${ZEALT_RUN_ID} (500x64-d float32) at container start
#   - /usr/local/bin/entrypoint.sh: seed-then-sleep entrypoint
