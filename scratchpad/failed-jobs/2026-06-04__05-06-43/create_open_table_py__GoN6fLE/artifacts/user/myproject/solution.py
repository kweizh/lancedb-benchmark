import os
import json
import lancedb
import pyarrow as pa
import numpy as np

def main():
    # Get URI
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    
    # Ensure output directory exists
    os.makedirs("/workspace/output", exist_ok=True)
    
    # Connect to db
    db = lancedb.connect(uri)
    
    # Create explicit PyArrow schema
    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("vector", pa.list_(pa.float32(), 4))
    ])
    
    # Generate 6 rows of data
    rng = np.random.default_rng(7)
    vectors = rng.random((6, 4)).astype(np.float32)
    
    data1 = [
        {"id": 1, "name": "Product A", "price": 10.99, "tags": ["tag1", "tag2"], "vector": vectors[0].tolist()},
        {"id": 2, "name": "Product B", "price": 15.50, "tags": ["tag3"], "vector": vectors[1].tolist()},
        {"id": 3, "name": "Product C", "price": 8.00, "tags": ["tag1", "tag4"], "vector": vectors[2].tolist()},
        {"id": 4, "name": "Product D", "price": 22.99, "tags": [], "vector": vectors[3].tolist()},
        {"id": 5, "name": "Product E", "price": 5.99, "tags": ["tag2", "tag3"], "vector": vectors[4].tolist()},
        {"id": 6, "name": "Product F", "price": 12.00, "tags": ["tag5"], "vector": vectors[5].tolist()}
    ]
    
    # Create table
    db.create_table("products", data=data1, schema=schema)
    
    # Different set of rows
    vectors2 = rng.random((3, 4)).astype(np.float32)
    data2 = [
        {"id": 7, "name": "Product G", "price": 1.99, "tags": ["tag6"], "vector": vectors2[0].tolist()},
        {"id": 8, "name": "Product H", "price": 2.99, "tags": ["tag7"], "vector": vectors2[1].tolist()},
        {"id": 9, "name": "Product I", "price": 3.99, "tags": ["tag8"], "vector": vectors2[2].tolist()}
    ]
    
    # Overwrite
    db.create_table("products", data=data2, schema=schema, mode="overwrite")
    
    # Restore original
    db.create_table("products", data=data1, schema=schema, mode="overwrite")
    
    # Reopen table
    table = db.open_table("products")
    
    # Get summary data
    tables_in_db = sorted(db.table_names())
    row_count = table.count_rows()
    schema_field_names = sorted(table.schema.names)
    
    summary = {
        "tables_in_db": tables_in_db,
        "row_count": row_count,
        "schema_field_names": schema_field_names
    }
    
    with open("/workspace/output/table_state.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
