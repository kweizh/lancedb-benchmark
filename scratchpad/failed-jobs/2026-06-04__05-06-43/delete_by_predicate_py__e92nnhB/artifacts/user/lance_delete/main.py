import os
import json
import pyarrow as pa
import lancedb

def main():
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(db_uri)
    
    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("level", pa.string()),
        pa.field("seq", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 4))
    ])
    
    # Generate data
    data = []
    levels = ["info", "warn", "error"]
    for i in range(1, 51):
        level = levels[(i - 1) % 3]
        seq = i * 2
        vector = [float(i), float(i), float(i), float(i)]
        data.append({"id": i, "level": level, "seq": seq, "vector": vector})
    
    # Create table
    # Drop if exists for idempotency
    if "logs" in db.table_names():
        db.drop_table("logs")
        
    tbl = db.create_table("logs", data=data, schema=schema)
    
    # Deletes
    tbl.delete("level = 'warn'")
    tbl.delete("level = 'info' AND seq > 60")
    tbl.delete("id IN (5, 9, 13)")
    
    # Fetch results
    total_rows = tbl.count_rows()
    df = tbl.to_pandas()
    remaining_ids_sorted = sorted(df["id"].tolist())
    
    # Output
    out_dir = "/workspace/output"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "delete_state.json")
    
    with open(out_file, "w") as f:
        json.dump({
            "total_rows": total_rows,
            "remaining_ids_sorted": remaining_ids_sorted
        }, f)

if __name__ == "__main__":
    main()
