import os
import numpy as np
import pyarrow as pa
import lancedb
from datetime import timedelta

# Resolve environment variables
db_path = os.environ.get("LANCE_DB_PATH")
run_id = os.environ.get("ZEALT_RUN_ID")

if not db_path:
    raise ValueError("LANCE_DB_PATH environment variable is not set")
if not run_id:
    raise ValueError("ZEALT_RUN_ID environment variable is not set")

table_name = f"vectors_{run_id}"

# Global connection and table reference
db = lancedb.connect(db_path)
table = None

def main():
    global table
    # Schema
    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 128))
    ])

    # Generate deterministic vectors
    rng = np.random.default_rng(2026)
    vecs = rng.standard_normal((1024, 128)).astype("float32")
    data = [{"id": i, "vector": vecs[i].tolist()} for i in range(1024)]

    # Create / Overwrite table
    table = db.create_table(table_name, data=data, schema=schema, mode="overwrite")

    # Create index
    table.create_index(
        metric="cosine",
        num_partitions=8,
        index_type="IVF_HNSW_SQ",
        replace=True
    )

    # Wait for index to be ready
    indices = table.list_indices()
    index_names = [idx.name for idx in indices]
    table.wait_for_index(index_names, timeout=timedelta(seconds=120))

# Execute main on import
main()

def search(query_vec, k, nprobes):
    """
    Runs a cosine vector search on the indexed table and varies nprobes per call.
    
    Parameters:
    - query_vec: list or array of 128 floats
    - k: number of results to return
    - nprobes: number of probes to use at query time
    
    Returns:
    - A list of dict rows in distance-ascending order; each row must contain at least the integer id field.
    """
    global table
    if table is None:
        table = db.open_table(table_name)
    results = table.search(query_vec).nprobes(nprobes).limit(k).to_list()
    return results
