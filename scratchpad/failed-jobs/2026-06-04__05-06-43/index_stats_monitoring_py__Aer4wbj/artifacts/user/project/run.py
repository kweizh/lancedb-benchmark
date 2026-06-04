import os
import json
import datetime
import pyarrow as pa
import numpy as np
import lancedb

def main():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)
    
    table_name = "points"
    if table_name in db.table_names():
        db.drop_table(table_name)
        
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])
    
    rng = np.random.default_rng(50)
    
    def generate_data(num_rows, start_id=0):
        data = []
        for i in range(num_rows):
            vec = rng.random(16).astype(np.float32).tolist()
            data.append({"id": start_id + i, "vector": vec})
        return data

    initial_data = generate_data(400, 0)
    table = db.create_table(table_name, data=initial_data, schema=schema)
    
    table.create_index(
        metric="cosine", 
        vector_column_name="vector", 
        index_type="IVF_PQ", 
        num_partitions=4, 
        num_sub_vectors=4, 
        replace=True
    )
    
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=120))
    
    stats1 = table.index_stats("vector_idx")
    
    added_data = generate_data(50, 400)
    table.add(added_data)
    
    stats2 = table.index_stats("vector_idx")
    
    os.makedirs("/workspace/output", exist_ok=True)
    
    output = {
        "index_type": str(stats1.index_type),
        "initial_indexed": int(stats1.num_indexed_rows),
        "initial_unindexed": int(stats1.num_unindexed_rows),
        "unindexed_after_add": int(stats2.num_unindexed_rows)
    }
    
    with open("/workspace/output/index_stats.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
