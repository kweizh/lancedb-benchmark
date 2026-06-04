import os
import datetime
import json
import numpy as np
import pyarrow as pa
import lancedb

def main():
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at: {uri}")
    db = lancedb.connect(uri)

    # 1. Define schema
    # Fixed size list of float32, size 32
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("tag", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 32))
    ])

    # 2. Seed the table with 512 deterministic rows
    rng = np.random.default_rng(2024)
    num_rows = 512
    vectors = rng.random((num_rows, 32), dtype=np.float32)
    ids = list(range(1, num_rows + 1))
    tags = [f"tag_{i}" for i in ids]

    # Convert vectors list to format compatible with schema
    vector_data = [v.tolist() for v in vectors]

    data_table = pa.Table.from_pydict({
        "id": ids,
        "tag": tags,
        "vector": vector_data
    }, schema=schema)

    print("Creating/overwriting table 'embeddings'...")
    table = db.create_table("embeddings", data=data_table, mode="overwrite")
    print(f"Table created. Row count: {len(table)}")

    # 3. Build IVF_PQ index
    print("Building IVF_PQ index on 'vector' column...")
    table.create_index(
        vector_column_name="vector",
        metric="cosine",
        num_partitions=4,
        num_sub_vectors=8,
        index_type="IVF_PQ",
        replace=True
    )

    # 4. Wait for the index to finish training
    print("Waiting for index to finish training...")
    # Use datetime.timedelta timeout
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=120))
    print("Index training complete.")

    # 5. Read the index stats
    indices = table.list_indices()
    print("List of indices:")
    index_present = False
    for idx in indices:
        print(f" - Name: {idx.name}, Type: {idx.index_type}, Columns: {idx.columns}")
        idx_type_upper = idx.index_type.upper()
        if "vector" in idx.columns and (idx_type_upper == "IVF_PQ" or idx_type_upper == "IVFPQ"):
            index_present = True

    try:
        stats = table.index_stats("vector_idx")
        print(f"Index stats: {stats}")
        # stats is an IndexStatistics object, we can access num_indexed_rows
        num_indexed_rows = stats.num_indexed_rows if stats else 0
    except Exception as e:
        print(f"Error getting index stats: {e}")
        num_indexed_rows = 0

    # 6. Run vector search for 10 nearest neighbors
    # Deterministic query vector generated with numpy.random.default_rng(99)
    query_rng = np.random.default_rng(99)
    query_vector = query_rng.random(32, dtype=np.float32)

    print("Running vector search...")
    search_res = table.search(query_vector).metric("cosine").limit(10).to_list()
    print(f"Search results: {search_res}")

    topk_ids = [row["id"] for row in search_res]
    print(f"Top-10 IDs: {topk_ids}")

    # 7. Write results to /workspace/output/ivf_pq.json
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ivf_pq.json")

    output_data = {
        "index_present": index_present,
        "num_indexed_rows": num_indexed_rows,
        "topk_ids": topk_ids
    }

    print(f"Writing output to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
