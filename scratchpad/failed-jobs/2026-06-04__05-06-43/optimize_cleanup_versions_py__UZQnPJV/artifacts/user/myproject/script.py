import os
import json
import lancedb
import pyarrow as pa
import numpy as np
from datetime import timedelta

# Connect to a LanceDB store
uri = os.environ.get("LANCEDB_URI", "/workspace/db")
db = lancedb.connect(uri)

# Define schema
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("value", pa.float32()),
    pa.field("vector", pa.list_(pa.float32(), 8))
])

# Seed the table with 100 rows
def generate_data(start_id, count):
    return [
        {
            "id": i,
            "value": float(i * 0.1),
            "vector": np.random.rand(8).astype(np.float32).tolist()
        }
        for i in range(start_id, start_id + count)
    ]

seed_data = generate_data(0, 100)
table = db.create_table("metrics", data=seed_data, schema=schema, mode="overwrite")

# Perform 8 small table.add calls of 10 rows each
current_id = 100
for _ in range(8):
    batch_data = generate_data(current_id, 10)
    table.add(batch_data)
    current_id += 10

# Capture pre_optimize_versions
pre_optimize_versions = len(table.list_versions())

# Call table.optimize
table.optimize(cleanup_older_than=timedelta(seconds=0))

# Capture post_optimize_versions
post_optimize_versions = len(table.list_versions())

# Capture post_optimize_row_count
post_optimize_row_count = table.count_rows()

# Write the result JSON
output_path = "/workspace/output/optimize_state.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump({
        "pre_optimize_versions": pre_optimize_versions,
        "post_optimize_versions": post_optimize_versions,
        "post_optimize_row_count": post_optimize_row_count
    }, f)

print(f"Results written to {output_path}")
