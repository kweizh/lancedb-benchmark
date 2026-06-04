import os
import json
import lancedb
import pyarrow as pa
import numpy as np
import datetime

def main():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)
    
    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("tag", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 32))
    ])
    
    # Generate data
    rng = np.random.default_rng(2024)
    num_rows = 512
    vectors = rng.random((num_rows, 32), dtype=np.float32)
    
    data = []
    for i in range(num_rows):
        data.append({
            "id": i + 1,
            "tag": f"tag_{i+1}",
            "vector": vectors[i].tolist()
        })
        
    # Create table
    table = db.create_table("embeddings", data=data, schema=schema, mode="overwrite")
    
    # Create index
    table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=8, replace=True)
    
    # Wait for index
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=120))
    
    # Check index present
    indices = table.list_indices()
    index_present = False
    for idx in indices:
        if "vector" in idx.columns:
            if "IVFPQ" in str(idx.index_type).upper() or "IVF_PQ" in str(idx.index_type).upper():
                index_present = True
                break
                
    # Get stats
    stats = table.index_stats("vector_idx")
    num_indexed_rows = stats.num_indexed_rows
    
    # Search
    query_rng = np.random.default_rng(99)
    query_vector = query_rng.random(32, dtype=np.float32)
    
    results = table.search(query_vector).limit(10).to_list()
    topk_ids = [res["id"] for res in results]
    
    # Write output
    os.makedirs("/workspace/output", exist_ok=True)
    output = {
        "index_present": bool(index_present),
        "num_indexed_rows": int(num_indexed_rows),
        "topk_ids": [int(i) for i in topk_ids]
    }
    
    with open("/workspace/output/ivf_pq.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
