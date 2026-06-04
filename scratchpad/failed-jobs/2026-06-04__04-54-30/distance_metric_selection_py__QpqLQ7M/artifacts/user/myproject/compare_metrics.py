import os
import json
import lancedb
import pyarrow as pa
import numpy as np

def main():
    # 1. Connect to LanceDB using URI from environment variable
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at: {uri}")
    db = lancedb.connect(uri)

    # 2. Define Arrow schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("label", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])

    # 3. Seed table with 32 deterministic rows from numpy.random.default_rng(123)
    rng = np.random.default_rng(123)
    vectors_data = rng.standard_normal(size=(32, 16)).astype("float32")
    
    # 4. Generate query vector from the same RNG instance
    query_vector = rng.standard_normal(size=(16,)).astype("float32")

    # Construct PyArrow arrays
    ids_array = pa.array(range(32), type=pa.int64())
    labels_array = pa.array([f"item-{i}" for i in range(32)], type=pa.string())
    vectors_array = pa.array(vectors_data.tolist(), type=pa.list_(pa.float32(), 16))

    # Create PyArrow table
    table_data = pa.Table.from_arrays(
        [ids_array, labels_array, vectors_array],
        schema=schema
    )

    # Create / Overwrite vectors table
    print("Creating/Overwriting table 'vectors'...")
    table = db.create_table("vectors", data=table_data, mode="overwrite")

    # 5. Run vector searches for l2, cosine, and dot
    results = {}
    for metric in ["l2", "cosine", "dot"]:
        search_res = table.search(query_vector).distance_type(metric).limit(5).to_arrow()
        ids = [int(x) for x in search_res["id"].to_pylist()]
        results[metric] = ids
        print(f"Distance metric '{metric}' top-5 IDs: {ids}")

    # 6. Ensure /workspace/output directory exists
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)

    # Write results to distances.json
    output_path = os.path.join(output_dir, "distances.json")
    print(f"Writing results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done successfully!")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Use os._exit(0) to avoid PyGILState_Release thread state crashes during Python finalization
        os._exit(0)
