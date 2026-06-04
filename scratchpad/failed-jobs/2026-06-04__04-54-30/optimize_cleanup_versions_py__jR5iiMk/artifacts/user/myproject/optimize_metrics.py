import os
import json
from datetime import timedelta
import lancedb
import pyarrow as pa
import numpy as np

def main():
    # 1. Connect to LanceDB
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at {db_uri}...")
    db = lancedb.connect(db_uri)
    
    # Clean up table if it already exists to ensure a clean run
    if "metrics" in db.table_names():
        db.drop_table("metrics")
        
    # 2. Define schema
    # id: int64
    # value: float32
    # vector: fixed_size_list<float32>[8]
    schema = pa.schema([
        ("id", pa.int64()),
        ("value", pa.float32()),
        ("vector", pa.list_(pa.float32(), 8))
    ])
    
    # 3. Seed table with 100 rows
    print("Seeding table with 100 rows...")
    ids = np.arange(100, dtype=np.int64)
    values = np.random.rand(100).astype(np.float32)
    vectors = np.random.rand(100, 8).astype(np.float32).tolist()
    
    seed_data = pa.Table.from_pydict({
        "id": ids,
        "value": values,
        "vector": vectors
    }, schema=schema)
    
    table = db.create_table("metrics", schema=schema, data=seed_data)
    
    # 4. Perform 8 small add calls of 10 rows each
    for i in range(8):
        print(f"Adding batch {i+1}/8 of 10 rows...")
        start_id = 100 + i * 10
        batch_ids = np.arange(start_id, start_id + 10, dtype=np.int64)
        batch_values = np.random.rand(10).astype(np.float32)
        batch_vectors = np.random.rand(10, 8).astype(np.float32).tolist()
        
        batch_data = pa.Table.from_pydict({
            "id": batch_ids,
            "value": batch_values,
            "vector": batch_vectors
        }, schema=schema)
        
        table.add(batch_data)
        
    # 5. Capture pre-optimize versions
    pre_versions = len(table.list_versions())
    print(f"Pre-optimize versions count: {pre_versions}")
    
    # 6. Call table.optimize(cleanup_older_than=timedelta(seconds=0))
    print("Running optimize with cleanup_older_than=timedelta(seconds=0)...")
    table.optimize(cleanup_older_than=timedelta(seconds=0))
    
    # 7. Capture post-optimize versions and row count
    post_versions = len(table.list_versions())
    post_row_count = table.count_rows()
    print(f"Post-optimize versions count: {post_versions}")
    print(f"Post-optimize row count: {post_row_count}")
    
    # 8. Write output to /workspace/output/optimize_state.json
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "optimize_state.json")
    
    result = {
        "pre_optimize_versions": pre_versions,
        "post_optimize_versions": post_versions,
        "post_optimize_row_count": post_row_count
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {output_path}: {result}")

if __name__ == "__main__":
    main()
