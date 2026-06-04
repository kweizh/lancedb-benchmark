import os
import json
import pyarrow as pa
import lancedb
import numpy as np

def main():
    # Connect to LanceDB
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(db_uri)
    
    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("label", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])
    
    # Generate data deterministically
    rng = np.random.default_rng(123)
    vectors = rng.standard_normal(size=(32, 16)).astype("float32")
    
    data = []
    for i in range(32):
        data.append({
            "id": i,
            "label": f"item-{i}",
            "vector": vectors[i].tolist()
        })
        
    # Drop if exists
    if "vectors" in db.table_names():
        db.drop_table("vectors")
        
    table = db.create_table("vectors", data=data, schema=schema)
    
    # Generate query vector
    query_vector = rng.standard_normal(size=(16,)).astype("float32")
    
    # Run vector searches
    results = {}
    for metric in ["l2", "cosine", "dot"]:
        search_res = table.search(query_vector).distance_type(metric).limit(5).to_list()
        results[metric] = [row["id"] for row in search_res]
        
    # Write to output file
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "distances.json")
    with open(output_path, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
