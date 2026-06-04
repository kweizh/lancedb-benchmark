import os
import json
import numpy as np
import pyarrow as pa
import lancedb

def main():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)
    
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 12))
    ])
    
    rng = np.random.default_rng(33)
    
    data = []
    for i in range(64):
        vec = rng.random(12, dtype=np.float32)
        data.append({
            "id": i,
            "name": f"item-{i}",
            "vector": vec
        })
    
    # Create the table
    if "items" in db.table_names():
        db.drop_table("items")
    tbl = db.create_table("items", schema=schema, data=data)
    
    # Generate 5 queries
    queries = []
    for _ in range(5):
        queries.append(rng.random(12, dtype=np.float32))
    
    queries_arr = np.stack(queries)
    
    # Run batched vector search
    try:
        res_df = tbl.search(queries_arr).limit(3).to_pandas()
        grouped = res_df.groupby("query_index")
        results = []
        for i in range(5):
            group = grouped.get_group(i)
            # Ensure the order is preserved (it should be sorted by distance by LanceDB)
            results.append(group["id"].tolist())
    except Exception as e:
        # Fallback if batched search fails or query_index is not present
        results = []
        for q in queries_arr:
            res = tbl.search(q).limit(3).to_list()
            results.append([r["id"] for r in res])
            
    # Write to output file
    os.makedirs("/workspace/output", exist_ok=True)
    with open("/workspace/output/batch_search.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

if __name__ == "__main__":
    main()
