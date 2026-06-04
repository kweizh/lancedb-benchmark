#!/usr/bin/env python3
"""Compare distance metrics in LanceDB vector search."""

import json
import os

import lancedb
import numpy as np
import pyarrow as pa

# --- Configuration ---
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "vectors"
DIM = 16
NUM_ROWS = 32
SEED = 123
TOP_K = 5
OUTPUT_PATH = "/workspace/output/distances.json"

# --- Deterministic data generation ---
rng = np.random.default_rng(SEED)

# Generate 32 vectors of dimension 16
vectors = rng.standard_normal(size=(NUM_ROWS, DIM)).astype("float32")

# Generate the query vector from the SAME rng instance (after generating data vectors)
query_vector = rng.standard_normal(size=(DIM,)).astype("float32")

# --- Build Arrow table ---
ids = pa.array(range(NUM_ROWS), type=pa.int64())
labels = pa.array([f"item-{i}" for i in range(NUM_ROWS)], type=pa.string())

# Build FixedSizeListArray from a flat values array
flat_values = pa.array(vectors.flatten(), type=pa.float32())
vectors_list = pa.FixedSizeListArray.from_arrays(flat_values, DIM)

schema = pa.schema([
    ("id", pa.int64()),
    ("label", pa.string()),
    ("vector", pa.list_(pa.float32(), DIM)),
])

table = pa.table(
    {"id": ids, "label": labels, "vector": vectors_list},
    schema=schema,
)

# --- Connect to LanceDB and create table ---
db = lancedb.connect(LANCEDB_URI)
db.drop_table(TABLE_NAME, ignore_missing=True)
tbl = db.create_table(TABLE_NAME, table)

# --- Run vector searches with three distance metrics ---
results = {}

for metric in ["l2", "cosine", "dot"]:
    search_result = tbl.search(query_vector).distance_type(metric).limit(TOP_K).to_arrow()
    top_ids = search_result.column("id").to_pylist()
    results[metric] = top_ids

# --- Write output ---
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results written to {OUTPUT_PATH}")
print(json.dumps(results, indent=2))