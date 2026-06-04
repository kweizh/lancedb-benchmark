FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        flask \
        "lancedb==0.25.3" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pandas==2.2.3" \
        pytest-xprocess \
        requests \
        beautifulsoup4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /home/user/myproject/data

# Inline build-time seed script: creates a LanceDB `products` table with 5
# deterministic rows (32-d random vectors from a fixed numpy seed) and writes
# the build-time fixture used by the verifier.
RUN cat > /tmp/seed.py <<'PY'
import json
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

DB_DIR = "/home/user/myproject/data/lancedb"
FIXTURE_PATH = "/home/user/myproject/.expected.json"
TABLE_NAME = "products"
VECTOR_DIM = 32

SEEDED = [
    {"id": "prod-001", "name": "Acme Notebook",    "category": "stationery", "price": 4.99},
    {"id": "prod-002", "name": "Acme Pen Pack",    "category": "stationery", "price": 2.49},
    {"id": "prod-003", "name": "Acme Desk Lamp",   "category": "lighting",   "price": 24.95},
    {"id": "prod-004", "name": "Acme USB Cable",   "category": "electronics", "price": 9.95},
    {"id": "prod-005", "name": "Acme Coffee Mug",  "category": "kitchenware", "price": 12.5},
]

def main():
    rng = np.random.default_rng(20260604)
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("category", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])
    rows = []
    for entry in SEEDED:
        vec = rng.standard_normal(VECTOR_DIM).astype(np.float32).tolist()
        rows.append({
            "id": entry["id"],
            "name": entry["name"],
            "category": entry["category"],
            "price": float(entry["price"]),
            "vector": vec,
        })

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=rows, schema=schema)
    assert tbl.count_rows() == len(rows), "Seed row count mismatch"

    fixture = {"seeded_products": SEEDED, "vector_dim": VECTOR_DIM}
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print("[seed] wrote fixture to", FIXTURE_PATH)

    # Work around lancedb 0.25.3 occasional SIGABRT during interpreter shutdown
    # in write-heavy build-time seed scripts.
    os._exit(0)

main()
PY

RUN python3 /tmp/seed.py && rm -f /tmp/seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="Flask CRUD admin dashboard over a pre-seeded LanceDB products table (5 rows, 32-d vectors)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, flask, lancedb==0.25.3, pyarrow==17.0.0,
#           numpy==1.26.4, pandas==2.2.3, pytest-xprocess, requests, beautifulsoup4
#   - /home/user/myproject: candidate project root
#   - /home/user/myproject/data/lancedb: pre-seeded LanceDB store with 5 product rows
#   - /home/user/myproject/.expected.json: build-time fixture (seeded_products list)
