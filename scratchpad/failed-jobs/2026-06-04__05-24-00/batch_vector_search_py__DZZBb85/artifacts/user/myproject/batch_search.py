import os
import json
import numpy as np
import pyarrow as pa
import lancedb

# --- Configuration ---
URI = os.environ.get("LANCEDB_URI", "/workspace/db")

# --- Deterministic RNG ---
rng = np.random.default_rng(33)

# --- Build Arrow schema ---
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("name", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 12)),
])

# --- Seed 64 rows ---
ids = []
names = []
vectors = []
for i in range(64):
    ids.append(i)
    names.append(f"item-{i}")
    vectors.append(rng.random(12, dtype=np.float32).tolist())

rows = pa.table({
    "id": pa.array(ids, type=pa.int64()),
    "name": pa.array(names, type=pa.string()),
    "vector": pa.array(vectors, type=pa.list_(pa.float32(), 12)),
}, schema=schema)

# --- Connect and create table ---
db = lancedb.connect(URI)
tbl = db.create_table("items", data=rows, mode="overwrite")

# --- Build 5 query vectors (draws 65..69 from the same RNG) ---
queries = np.stack([rng.random(12, dtype=np.float32) for _ in range(5)])  # shape (5, 12)

# --- Batch vector search ---
results = tbl.search(queries).limit(3).to_pandas()

# --- Extract top-3 ids per query ---
output_results = []
for qi in range(5):
    subset = results[results["query_index"] == qi]
    top_ids = subset["id"].tolist()[:3]
    output_results.append(top_ids)

# --- Write output ---
os.makedirs("/workspace/output", exist_ok=True)
with open("/workspace/output/batch_search.json", "w") as f:
    json.dump({"results": output_results}, f)

print("Done. Results written to /workspace/output/batch_search.json")