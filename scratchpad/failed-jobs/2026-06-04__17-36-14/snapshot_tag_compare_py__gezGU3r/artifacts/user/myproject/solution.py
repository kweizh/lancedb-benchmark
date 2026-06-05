import os
import numpy as np
import pyarrow as pa
import lancedb

def build_snapshots(db_path: str, table_name: str) -> None:
    # Initialize RNG
    rng = np.random.default_rng(seed=2026)
    
    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # --- Seed phase (write version v1) ---
    # Generate 50 rows with id in [0, 50)
    ids_v1 = list(range(50))
    texts_v1 = [f"doc-{i}" for i in ids_v1]
    vectors_v1 = rng.random((50, 16), dtype=np.float32)
    
    arr_v1 = pa.FixedSizeListArray.from_arrays(vectors_v1.flatten(), 16)
    table_v1 = pa.Table.from_arrays([
        pa.array(ids_v1, type=pa.int64()),
        pa.array(texts_v1, type=pa.string()),
        arr_v1
    ], schema=schema)
    
    # Create the table with initial data (this creates version 1)
    tbl = db.create_table(table_name, data=table_v1, schema=schema, mode='overwrite')
    
    # Attach tag v1_baseline
    tbl.tags.create("v1_baseline", tbl.version)
    
    # --- Extend phase (write version v2) ---
    # Append 20 more rows with id in [50, 70)
    ids_v2 = list(range(50, 70))
    texts_v2 = [f"doc-{i}" for i in ids_v2]
    vectors_v2 = rng.random((20, 16), dtype=np.float32)
    
    arr_v2 = pa.FixedSizeListArray.from_arrays(vectors_v2.flatten(), 16)
    table_v2 = pa.Table.from_arrays([
        pa.array(ids_v2, type=pa.int64()),
        pa.array(texts_v2, type=pa.string()),
        arr_v2
    ], schema=schema)
    
    # Append data (this creates version 2)
    tbl.add(table_v2)
    
    # Attach tag v2_extended
    tbl.tags.create("v2_extended", tbl.version)
    
    # --- Prune phase (write version v3) ---
    # Delete rows with id < 5 (this creates version 3)
    tbl.delete("id < 5")
    
    # Attach tag v3_pruned
    tbl.tags.create("v3_pruned", tbl.version)

def diff(db_path: str, table_name: str, tag_a: str, tag_b: str) -> dict:
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    tbl = db.open_table(table_name)
    
    try:
        # Read ids from snapshot A
        tbl.checkout(tag_a)
        ids_a = set(tbl.search().select(["id"]).to_arrow()["id"].to_pylist())
        
        # Read ids from snapshot B
        tbl.checkout(tag_b)
        ids_b = set(tbl.search().select(["id"]).to_arrow()["id"].to_pylist())
        
        # Calculate added, removed, and common IDs
        added_ids = sorted(list(ids_b - ids_a))
        removed_ids = sorted(list(ids_a - ids_b))
        common_count = len(ids_a & ids_b)
        
        return {
            "added_ids": added_ids,
            "removed_ids": removed_ids,
            "common_count": common_count
        }
    finally:
        # Always checkout latest version to leave the table in the live state
        tbl.checkout_latest()

if __name__ == "__main__":
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    db_path = "/app/db"
    table_name = f"documents_{run_id}"
    
    print(f"Building snapshots for table '{table_name}' in database '{db_path}'...")
    build_snapshots(db_path, table_name)
    print("Snapshots built successfully!")
