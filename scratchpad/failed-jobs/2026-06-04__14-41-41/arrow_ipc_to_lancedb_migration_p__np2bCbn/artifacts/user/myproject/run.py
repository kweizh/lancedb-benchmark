import os
import sys
import json
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import lancedb

def main():
    # Retrieve ZEALT_RUN_ID from environment
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("Error: ZEALT_RUN_ID environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    
    table_name = f"events_{run_id}"
    db_path = "/home/user/myproject/lancedb_data"
    arrows_path = "/app/source/dataset.arrows"
    query_vector_path = "/app/query_vector.npy"
    
    # 1. Read the Arrow IPC stream
    if not os.path.exists(arrows_path):
        print(f"Error: Arrow IPC stream not found at {arrows_path}", file=sys.stderr)
        sys.exit(1)
        
    with ipc.open_stream(arrows_path) as reader:
        source_schema = reader.schema
        table = reader.read_all()
        
    row_count = table.num_rows
    
    # 2. Connect to LanceDB and create/overwrite table
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    
    # Create/overwrite the table with the exact schema and data
    tbl = db.create_table(table_name, data=table, mode="overwrite")
    
    # 3. Verify schema parity
    schema_match = source_schema.equals(tbl.schema, check_metadata=False)
    
    # 4. Load query vector and run top-5 search
    if not os.path.exists(query_vector_path):
        print(f"Error: Query vector not found at {query_vector_path}", file=sys.stderr)
        sys.exit(1)
        
    query_vector = np.load(query_vector_path)
    
    # Search top 5 nearest-neighbors using L2 distance
    search_results = tbl.search(query_vector).metric("l2").limit(5).to_list()
    
    # Format top5 list
    top5 = []
    for r in search_results:
        top5.append({
            "id": int(r["id"]),
            "distance": float(r["_distance"])
        })
        
    # 5. Construct final JSON output
    output = {
        "table_name": table_name,
        "row_count": row_count,
        "schema_match": schema_match,
        "top5": top5
    }
    
    # Print exactly one JSON document to stdout
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
