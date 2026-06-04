import os
import json
import pyarrow as pa
import lancedb

def main():
    # Connect to LanceDB using URI from LANCEDB_URI (default: /workspace/db)
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at: {db_uri}")
    
    # Ensure parent directory of database exists
    db_dir = os.path.dirname(db_uri)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    db = lancedb.connect(db_uri)
    
    # Define the PyArrow schema
    schema = pa.schema([
        ("id", pa.int64()),
        ("level", pa.string()),
        ("seq", pa.int32()),
        ("vector", pa.list_(pa.float32(), 4))
    ])
    
    # Drop table if exists to ensure clean run
    table_name = "logs"
    if table_name in db.table_names():
        print(f"Table '{table_name}' already exists, dropping it...")
        db.drop_table(table_name)
        
    # Create the table
    tbl = db.create_table(table_name, schema=schema)
    print(f"Created empty table '{table_name}'.")
    
    # Seed the table with exactly 50 deterministic rows
    data = []
    levels = ["info", "warn", "error"]
    for i in range(1, 51):
        lvl = levels[(i - 1) % 3]
        seq = i * 2
        # Simple deterministic 4-float vector derived from i
        vec = [float(i), float(i + 1), float(i + 2), float(i + 3)]
        data.append({
            "id": i,
            "level": lvl,
            "seq": seq,
            "vector": vec
        })
        
    pa_table = pa.Table.from_pylist(data, schema=schema)
    tbl.add(pa_table)
    print(f"Seeded table with {len(data)} rows.")
    
    # Verify initial row count
    initial_count = tbl.count_rows()
    print(f"Initial row count: {initial_count}")
    assert initial_count == 50, f"Expected 50 rows, got {initial_count}"
    
    # Run the three deletes IN ORDER
    # 1. Delete every row whose level equals 'warn'
    print("Running Delete 1: level = 'warn'")
    tbl.delete("level = 'warn'")
    print(f"Row count after Delete 1: {tbl.count_rows()}")
    
    # 2. Delete every row whose level equals 'info' AND whose seq is strictly greater than 60
    print("Running Delete 2: level = 'info' AND seq > 60")
    tbl.delete("level = 'info' AND seq > 60")
    print(f"Row count after Delete 2: {tbl.count_rows()}")
    
    # 3. Delete every row whose id is in the set {5, 9, 13}
    print("Running Delete 3: id IN (5, 9, 13)")
    tbl.delete("id IN (5, 9, 13)")
    print(f"Row count after Delete 3: {tbl.count_rows()}")
    
    # Get surviving ids and sort them
    remaining_tbl = tbl.to_arrow()
    remaining_ids = remaining_tbl.column("id").to_pylist()
    remaining_ids_sorted = sorted(remaining_ids)
    total_rows = tbl.count_rows()
    
    print(f"Remaining rows (from count_rows): {total_rows}")
    print(f"Remaining rows (from list len): {len(remaining_ids_sorted)}")
    print(f"Remaining IDs: {remaining_ids_sorted}")
    
    # Write JSON output to /workspace/output/delete_state.json
    output_path = "/workspace/output/delete_state.json"
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_data = {
        "total_rows": total_rows,
        "remaining_ids_sorted": remaining_ids_sorted
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Successfully wrote output to {output_path}")

if __name__ == "__main__":
    main()
