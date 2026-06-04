#!/usr/bin/env python3
import os
import json
import numpy as np
import pyarrow as pa
import lancedb

def main():
    # 1. Connect to local LanceDB database
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to database at: {uri}")
    db = lancedb.connect(uri)

    # 2. Define the explicit PyArrow schema
    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("vector", pa.list_(pa.float32(), 4))
    ])
    print("Schema defined successfully.")

    # 3. Seed deterministic data (at least 6 rows)
    rng = np.random.default_rng(7)
    vectors_orig = [rng.random(4).tolist() for _ in range(6)]
    
    original_data = [
        {"id": 1, "name": "Apple", "price": 1.20, "tags": ["fruit", "red"], "vector": vectors_orig[0]},
        {"id": 2, "name": "Banana", "price": 0.50, "tags": ["fruit", "yellow"], "vector": vectors_orig[1]},
        {"id": 3, "name": "Cherry", "price": 3.00, "tags": ["fruit", "red"], "vector": vectors_orig[2]},
        {"id": 4, "name": "Date", "price": 2.50, "tags": ["fruit", "brown"], "vector": vectors_orig[3]},
        {"id": 5, "name": "Elderberry", "price": 4.00, "tags": ["fruit", "purple"], "vector": vectors_orig[4]},
        {"id": 6, "name": "Fig", "price": 1.80, "tags": ["fruit", "purple"], "vector": vectors_orig[5]},
    ]
    print("Original deterministic dataset prepared.")

    # 4. Demonstrate mode="overwrite" semantics
    # Create the table initially with original data
    print("Creating table 'products' with original data...")
    db.create_table("products", data=original_data, schema=schema, mode="overwrite")

    # Recreate/overwrite with a different but schema-compatible set of rows
    # We will generate 3 different rows
    vectors_diff = [rng.random(4).tolist() for _ in range(3)]
    different_data = [
        {"id": 10, "name": "Grape", "price": 2.00, "tags": ["fruit", "green"], "vector": vectors_diff[0]},
        {"id": 11, "name": "Honeydew", "price": 3.50, "tags": ["fruit", "green"], "vector": vectors_diff[1]},
        {"id": 12, "name": "Kiwi", "price": 1.50, "tags": ["fruit", "brown"], "vector": vectors_diff[2]},
    ]
    print("Overwriting table 'products' with different schema-compatible data...")
    db.create_table("products", data=different_data, schema=schema, mode="overwrite")

    # Restore the original 6-row dataset again via mode="overwrite"
    print("Restoring original 6-row dataset via mode='overwrite'...")
    db.create_table("products", data=original_data, schema=schema, mode="overwrite")

    # 5. Reopen the table
    print("Reopening table 'products'...")
    table = db.open_table("products")
    row_count = table.count_rows()
    print(f"Reopened table has {row_count} rows.")

    # 6. Gather information for JSON summary
    tables_in_db = sorted(db.table_names())
    schema_field_names = sorted(table.schema.names)

    summary = {
        "tables_in_db": tables_in_db,
        "row_count": row_count,
        "schema_field_names": schema_field_names
    }

    # Ensure output directory exists
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "table_state.json")

    # Write JSON summary
    print(f"Writing summary to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
