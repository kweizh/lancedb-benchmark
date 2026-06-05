FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "lancedb==0.25.3" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pandas==2.2.3" \
        "grpcio>=1.62,<2.0" \
        "grpcio-tools>=1.62,<2.0" \
        "protobuf>=4.25,<6.0" \
        "pytest-xprocess" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /home/user/myproject/data

# Inline build-time seed script: deterministically generates 500 rows of 24-d
# vectors and 4-way categorical metadata, writes them to a LanceDB table named
# 'documents', and pre-computes the expected ground-truth top-5 retrievals
# (with and without a category filter) for the fixture query vector.
RUN cat > /tmp/seed.py <<'PY'
import json
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

DB_DIR = "/home/user/myproject/data/lancedb"
FIXTURE_PATH = "/home/user/myproject/.expected.json"
TABLE_NAME = "documents"
NUM_ROWS = 500
DIM = 24
CATEGORIES = ["alpha", "beta", "gamma", "delta"]


def main():
    rng = np.random.default_rng(2026)
    vecs = rng.standard_normal(size=(NUM_ROWS, DIM)).astype(np.float32)
    ids = list(range(NUM_ROWS))
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in ids]
    titles = [f"doc_{i:03d}" for i in ids]

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("category", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ])
    rows = [
        {
            "id": int(ids[i]),
            "title": titles[i],
            "category": categories[i],
            "vector": vecs[i].tolist(),
        }
        for i in range(NUM_ROWS)
    ]

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=rows, schema=schema)
    assert tbl.count_rows() == NUM_ROWS, "Seed row count mismatch"

    # Pick query_vector as row 200 (category 'alpha') verbatim so the top-1
    # match is the row itself (distance 0) in both filtered and unfiltered runs.
    pivot = 200
    assert categories[pivot] == "alpha", (
        f"Expected pivot row {pivot} to be in category 'alpha', got {categories[pivot]}"
    )
    query_vector = vecs[pivot].astype(np.float32).tolist()

    # Compute deterministic ground-truth top-5 for the unfiltered query.
    no_filter_hits = tbl.search(query_vector).limit(5).to_list()
    top_k_no_filter = [{"id": int(h["id"]), "title": h["title"]} for h in no_filter_hits]
    assert len(top_k_no_filter) == 5, "Expected 5 hits for unfiltered top-5"
    assert top_k_no_filter[0]["id"] == pivot, (
        f"Expected pivot id {pivot} to be rank-1 unfiltered, got {top_k_no_filter[0]['id']}"
    )

    # Compute deterministic ground-truth top-5 with where clause.
    where_clause = "category = 'alpha'"
    filtered_hits = (
        tbl.search(query_vector).where(where_clause).limit(5).to_list()
    )
    top_k_with_filter = [{"id": int(h["id"]), "title": h["title"]} for h in filtered_hits]
    assert len(top_k_with_filter) == 5, "Expected 5 hits for filtered top-5"
    assert top_k_with_filter[0]["id"] == pivot, (
        f"Expected pivot id {pivot} to be rank-1 filtered, got {top_k_with_filter[0]['id']}"
    )
    for h in top_k_with_filter:
        assert h["id"] % len(CATEGORIES) == 0, (
            f"Filtered hit id {h['id']} is not an 'alpha' row (id % 4 != 0)."
        )

    # Sanity-check distance margins so reruns of the verifier are stable
    # against tiny floating-point reorderings.
    no_filter_6 = tbl.search(query_vector).limit(6).to_list()
    if len(no_filter_6) == 6:
        margin = float(no_filter_6[5]["_distance"]) - float(no_filter_6[4]["_distance"])
        print(f"[seed] unfiltered top-5/6 margin: {margin:.6f}")
        assert margin > 1e-4, "Unfiltered top-5 margin too small for stable verification"

    filtered_6 = tbl.search(query_vector).where(where_clause).limit(6).to_list()
    if len(filtered_6) == 6:
        margin_f = float(filtered_6[5]["_distance"]) - float(filtered_6[4]["_distance"])
        print(f"[seed] filtered  top-5/6 margin: {margin_f:.6f}")
        assert margin_f > 1e-4, "Filtered top-5 margin too small for stable verification"

    fixture = {
        "query_vector": query_vector,
        "top_k_no_filter": top_k_no_filter,
        "top_k_with_filter": top_k_with_filter,
        "where_clause": where_clause,
    }
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print("[seed] wrote fixture to", FIXTURE_PATH)
    print("[seed] top_k_no_filter:", top_k_no_filter)
    print("[seed] top_k_with_filter:", top_k_with_filter)

    # Avoid lancedb 0.25.3 occasional SIGABRT during interpreter shutdown in
    # write-heavy build-time seed scripts.
    os._exit(0)


main()
PY

RUN python3 /tmp/seed.py && rm -f /tmp/seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="gRPC vector-search microservice over a pre-seeded 500-row, 24-d LanceDB documents table."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3,
#           grpcio>=1.62, grpcio-tools>=1.62, protobuf>=4.25, pytest-xprocess
#   - /home/user/myproject: candidate project root (empty of candidate artifacts)
#   - /home/user/myproject/data/lancedb: pre-seeded 500-row, 24-d 'documents' table
#       schema: {id int64, title string, category string, vector fixed_size_list<float32, 24>}
#   - /home/user/myproject/.expected.json: ground-truth fixture (query_vector + top-5 with/without filter)
