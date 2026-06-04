import os
import json
import numpy as np
import pyarrow as pa
import lancedb

def main():
    # Get the LanceDB URI from the environment variable
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at: {uri}")
    
    # Connect to LanceDB
    db = lancedb.connect(uri)
    
    # Define the schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 12))
    ])
    
    # Create the items table
    tbl = db.create_table("items", schema=schema, mode="overwrite")
    
    # Initialize the RNG
    rng = np.random.default_rng(33)
    
    # Generate 64 rows
    data = []
    for i in range(64):
        v = rng.random(12, dtype=np.float32)
        data.append({
            "id": i,
            "name": f"item-{i}",
            "vector": v.tolist()
        })
        
    # Add the rows to the table
    tbl.add(data)
    print("Seeded 'items' table with 64 rows.")
    
    # Generate 5 query vectors using the same RNG
    queries = [rng.random(12, dtype=np.float32) for _ in range(5)]
    queries_arr = np.stack(queries)
    
    # Run the batched vector search
    results = []
    try:
        # Try the batched search first
        res = tbl.search(queries_arr).limit(3).to_pandas()
        # If the response has 'query_index', we extract based on it
        if "query_index" in res.columns:
            print("Using batched search query_index results.")
            for q in range(5):
                q_res = res[res["query_index"] == q]
                ids = [int(id_val) for id_val in q_res["id"].tolist()]
                results.append(ids)
        else:
            print("query_index not in columns, falling back to loop search.")
            # Fallback to loop if query_index is not present
            for q_vec in queries:
                search_res = tbl.search(q_vec).limit(3).to_list()
                ids = [int(row["id"]) for row in search_res]
                results.append(ids)
    except Exception as e:
        print(f"Batched search failed or raised exception: {e}. Falling back to loop search.")
        # Fallback to loop on error
        results = []
        for q_vec in queries:
            search_res = tbl.search(q_vec).limit(3).to_list()
            ids = [int(row["id"]) for row in search_res]
            results.append(ids)
            
    # Write the final answer to /workspace/output/batch_search.json
    os.makedirs("/workspace/output", exist_ok=True)
    output_path = "/workspace/output/batch_search.json"
    
    output_data = {
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Successfully wrote results to {output_path}")
    print(json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()
