"""
Build an IVF_PQ vector index in LanceDB, run a vector search, and
persist results to /workspace/output/ivf_pq.json.
"""

import datetime
import json
import os

import lancedb
import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ivf_pq.json")

NUM_ROWS = 512
DIM = 32
TABLE_NAME = "embeddings"
INDEX_NAME = "vector_idx"

# ---------------------------------------------------------------------------
# Step 1: Connect to LanceDB
# ---------------------------------------------------------------------------
print(f"Connecting to LanceDB at {LANCEDB_URI} ...")
db = lancedb.connect(LANCEDB_URI)

# ---------------------------------------------------------------------------
# Step 2: Build the table schema
# ---------------------------------------------------------------------------
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("tag", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

# ---------------------------------------------------------------------------
# Step 3: Generate deterministic data
# ---------------------------------------------------------------------------
print(f"Generating {NUM_ROWS} deterministic rows (seed=2024) ...")
rng = np.random.default_rng(2024)
vectors = rng.random((NUM_ROWS, DIM), dtype=np.float32)

ids = list(range(1, NUM_ROWS + 1))
tags = [f"tag_{i}" for i in ids]

table_data = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "tag": pa.array(tags, type=pa.utf8()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.flatten(), type=pa.float32()), DIM
        ),
    },
    schema=schema,
)

# ---------------------------------------------------------------------------
# Step 4: Create (or overwrite) the table
# ---------------------------------------------------------------------------
print(f"Creating / overwriting table '{TABLE_NAME}' ...")
table = db.create_table(TABLE_NAME, data=table_data, mode="overwrite")
print(f"  Row count: {table.count_rows()}")

# ---------------------------------------------------------------------------
# Step 5: Build IVF_PQ index
# ---------------------------------------------------------------------------
print("Building IVF_PQ index on 'vector' column ...")
table.create_index(
    "cosine",
    num_partitions=4,
    num_sub_vectors=8,
    vector_column_name="vector",
    index_type="IVF_PQ",
    replace=True,
)
print("  Index creation call returned.")

# ---------------------------------------------------------------------------
# Step 6: Wait for the index to finish training
# ---------------------------------------------------------------------------
print(f"Waiting for index '{INDEX_NAME}' to finish ...")
table.wait_for_index(
    [INDEX_NAME],
    timeout=datetime.timedelta(seconds=120),
)
print("  Index is ready.")

# ---------------------------------------------------------------------------
# Step 7: Verify the index is present
# ---------------------------------------------------------------------------
indices = table.list_indices()
print(f"  Indices on table: {indices}")

# index_type on IndexConfig is "IvfPq"; index_type on IndexStatistics is "IVF_PQ"
# Normalise by removing underscores and lower-casing for a robust comparison.
index_present = any(
    getattr(idx, "index_type", "").replace("_", "").lower() == "ivfpq"
    and getattr(idx, "columns", []) == ["vector"]
    for idx in indices
)
print(f"  IVF_PQ index present: {index_present}")

# ---------------------------------------------------------------------------
# Step 8: Read index stats
# ---------------------------------------------------------------------------
stats = table.index_stats(INDEX_NAME)
print(f"  Index stats: {stats}")

# IndexStatistics is a dataclass; fall back gracefully if API changes.
if isinstance(stats, dict):
    num_indexed_rows = stats.get("num_indexed_rows", stats.get("num_rows", 0))
else:
    num_indexed_rows = getattr(stats, "num_indexed_rows",
                               getattr(stats, "num_rows", 0))

print(f"  Indexed rows: {num_indexed_rows}")

# ---------------------------------------------------------------------------
# Step 9: Run a vector search (top-10 nearest neighbours)
# ---------------------------------------------------------------------------
query_rng = np.random.default_rng(99)
query_vec = query_rng.random(DIM, dtype=np.float32)

print("Running vector search (top-10) ...")
result_tbl = (
    table.search(query_vec)
    .limit(10)
    .to_arrow()
)
result_dict = result_tbl.to_pydict()
print(f"  Result ids: {result_dict['id']}")

topk_ids = [int(v) for v in result_dict["id"]]
print(f"  top-k ids: {topk_ids}")

# ---------------------------------------------------------------------------
# Step 10: Write output JSON
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

output = {
    "index_present": bool(index_present),
    "num_indexed_rows": int(num_indexed_rows),
    "topk_ids": topk_ids,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults written to {OUTPUT_FILE}")
print(json.dumps(output, indent=2))
