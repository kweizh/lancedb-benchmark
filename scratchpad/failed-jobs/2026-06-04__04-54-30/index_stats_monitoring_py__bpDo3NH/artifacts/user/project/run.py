import os
import json
import datetime
import numpy as np
import pyarrow as pa
import lancedb

def main():
    # 1. Connect to the LanceDB URI provided in the LANCEDB_URI environment variable
    lancedb_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at: {lancedb_uri}")
    db = lancedb.connect(lancedb_uri)

    # 2. Create a fresh table named 'points' (drop if exists for idempotency)
    table_name = "points"
    print(f"Ensuring clean '{table_name}' table exists...")
    db.drop_table(table_name, ignore_missing=True)

    # 3. Seed the table with exactly 400 deterministic rows generated from numpy.random.default_rng(50)
    print("Generating 400 seed vectors...")
    rng = np.random.default_rng(50)
    seed_vectors = rng.random((400, 16), dtype=np.float32)
    seed_ids = np.arange(400, dtype=np.int64)

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])

    seed_table = pa.Table.from_pydict({
        "id": seed_ids,
        "vector": list(seed_vectors)
    }, schema=schema)

    print("Creating and seeding table...")
    table = db.create_table(table_name, data=seed_table)

    # 4. Build an IVF_PQ index over the vector column
    print("Building IVF_PQ index...")
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        replace=True
    )

    print("Waiting for index to become ready...")
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=120))

    # 5. Capture an initial snapshot of table.index_stats("vector_idx")
    print("Capturing initial index stats...")
    stats_initial = table.index_stats("vector_idx")
    if stats_initial is None:
        raise ValueError("Index statistics not found after building index.")

    initial_indexed = int(stats_initial.num_indexed_rows)
    initial_unindexed = int(stats_initial.num_unindexed_rows)
    index_type = str(stats_initial.index_type)

    print(f"Initial Stats -> index_type: {index_type}, indexed: {initial_indexed}, unindexed: {initial_unindexed}")

    # 6. Append 50 additional deterministic rows to the table (must remain unindexed)
    print("Generating 50 additional vectors...")
    add_vectors = rng.random((50, 16), dtype=np.float32)
    add_ids = np.arange(400, 450, dtype=np.int64)

    add_table = pa.Table.from_pydict({
        "id": add_ids,
        "vector": list(add_vectors)
    }, schema=schema)

    print("Appending 50 additional rows...")
    table.add(add_table)

    # 7. Capture a second snapshot of table.index_stats("vector_idx")
    print("Capturing post-append index stats...")
    stats_after_add = table.index_stats("vector_idx")
    if stats_after_add is None:
        raise ValueError("Index statistics not found after appending rows.")

    unindexed_after_add = int(stats_after_add.num_unindexed_rows)

    print(f"Post-Add Stats -> unindexed: {unindexed_after_add}")

    # Write the monitoring report to /workspace/output/index_stats.json
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "index_stats.json")

    report = {
        "index_type": index_type,
        "initial_indexed": initial_indexed,
        "initial_unindexed": initial_unindexed,
        "unindexed_after_add": unindexed_after_add
    }

    print(f"Writing report to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Monitoring workflow completed successfully.")

if __name__ == "__main__":
    main()
