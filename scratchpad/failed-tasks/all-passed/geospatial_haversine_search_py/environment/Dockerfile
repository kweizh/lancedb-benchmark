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
RUN mkdir -p /home/user/myproject

RUN mkdir -p /opt/seed
RUN cat >/opt/seed/seed_pois.py <<'EOF'
"""Build-time seed script.

Generates the pre-seeded LanceDB POI corpus and the two query-vector files
that the candidate's `search.py` will consume.
"""
import json
import os

import lancedb
import numpy as np
import pyarrow as pa

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
TABLE_NAME = "pois"
N_ROWS = 240
SEED = 2026
CATEGORIES = ["restaurant", "cafe", "museum", "park", "shop"]


def build_dataset():
    rng = np.random.default_rng(SEED)
    lats = rng.uniform(37.0, 38.0, N_ROWS)
    lons = rng.uniform(-123.0, -122.0, N_ROWS)
    cats = rng.choice(CATEGORIES, N_ROWS)
    embeddings = rng.standard_normal((N_ROWS, 32)).astype(np.float32)
    ids = np.arange(N_ROWS, dtype=np.int32)
    names = [f"POI-{i:04d}" for i in ids]

    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("lat", pa.float64()),
        pa.field("lon", pa.float64()),
        pa.field("category", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 32)),
    ])

    arrays = [
        pa.array(ids, type=pa.int32()),
        pa.array(names, type=pa.string()),
        pa.array(lats, type=pa.float64()),
        pa.array(lons, type=pa.float64()),
        pa.array(cats, type=pa.string()),
        pa.array(list(embeddings), type=pa.list_(pa.float32(), 32)),
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def write_query_vector(path: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(32).astype(np.float32).tolist()
    with open(path, "w") as f:
        json.dump(vec, f)


def main():
    os.makedirs(PROJECT_DIR, exist_ok=True)
    table = build_dataset()
    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    db.create_table(TABLE_NAME, data=table)

    write_query_vector(os.path.join(PROJECT_DIR, "query_vec_A.json"), 2027)
    write_query_vector(os.path.join(PROJECT_DIR, "query_vec_B.json"), 2028)

    print(f"Seeded {N_ROWS} POIs into {DB_DIR}/{TABLE_NAME}")
    # Workaround for lancedb 0.25.3 SIGABRT on interpreter teardown after
    # write-heavy seed scripts.
    os._exit(0)


if __name__ == "__main__":
    main()
EOF

RUN python3 /opt/seed/seed_pois.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB POI corpus seeded with 240 deterministic geo+vector rows for the geospatial Haversine search task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy==1.26.4 + pandas==2.2.3 pinned for stable wheels on ubuntu:24.04
#   - /home/user/myproject/lancedb: pre-seeded LanceDB database with the 'pois' table (240 rows, 32-d embeddings)
#   - /home/user/myproject/query_vec_A.json, query_vec_B.json: deterministic 32-d query vectors used by the verifier
