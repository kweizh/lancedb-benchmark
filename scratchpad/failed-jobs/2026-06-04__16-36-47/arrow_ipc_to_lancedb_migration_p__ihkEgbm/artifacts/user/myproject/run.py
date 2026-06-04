import os
import json
import numpy as np
import pyarrow.ipc as ipc
import lancedb

def main():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"events_{run_id}"
    
    source_path = "/app/source/dataset.arrows"
    query_vector_path = "/app/query_vector.npy"
    db_path = "/home/user/myproject/lancedb_data"
    
    # Read the query vector
    query_vector = np.load(query_vector_path)
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # Read the IPC stream
    with ipc.open_stream(source_path) as reader:
        source_schema = reader.schema
        # Create table directly from the RecordBatchReader
        table = db.create_table(table_name, data=reader, mode="overwrite")
        
    # Verify row count
    row_count = table.count_rows()
    
    # Verify schema match
    schema_match = table.schema.equals(source_schema, check_metadata=False)
    
    # Search top 5
    results = table.search(query_vector).limit(5).to_arrow()
    
    # Extract results
    res_dict = results.to_pydict()
    top5 = []
    
    dist_col = "_distance"
    
    for i in range(len(res_dict["id"])):
        top5.append({
            "id": int(res_dict["id"][i]),
            "distance": float(res_dict[dist_col][i])
        })
        
    output = {
        "table_name": table_name,
        "row_count": row_count,
        "schema_match": schema_match,
        "top5": top5
    }
    
    print(json.dumps(output))

if __name__ == "__main__":
    main()
