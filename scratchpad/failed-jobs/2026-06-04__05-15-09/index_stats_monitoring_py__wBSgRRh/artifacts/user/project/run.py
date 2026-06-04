import os
import json
import datetime
import numpy as np
import pyarrow as pa
import lancedb

# Configuration
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "index_stats.json")
TABLE_NAME = "points"
VECTOR_DIM = 16
SEED_ROWS = 400
EXTRA_ROWS = 50
RNG_SEED = 50

def generate_rows(rng, n):
    ids = list(range(n))
    vectors = rng.random((n, VECTOR_DIM), dtype=np.float32).tolist()
    return [{"id": i, "vector": v} for i, v in zip(ids, vectors)]

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Connect to LanceDB
    db = lancedb.connect(LANCEDB_URI)

    # Drop existing table if present (idempotent)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 16)),
    ])

    # Seed table with 400 deterministic rows
    rng = np.random.default_rng(RNG_SEED)
    seed_data = generate_rows(rng, SEED_ROWS)

    table = db.create_table(TABLE_NAME, data=seed_data, schema=schema)
    print(f"Created table '{TABLE_NAME}' with {SEED_ROWS} rows.")

    # Build IVF_PQ index
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        replace=True,
    )
    print("Index creation initiated. Waiting for index to become ready...")

    # Wait for index to be ready
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=120))
    print("Index is ready.")

    # Capture initial snapshot
    stats_initial = table.index_stats("vector_idx")
    initial_indexed = int(stats_initial.num_indexed_rows)
    initial_unindexed = int(stats_initial.num_unindexed_rows)
    index_type = stats_initial.index_type
    print(f"Initial snapshot -> indexed: {initial_indexed}, unindexed: {initial_unindexed}, type: {index_type}")

    # Append 50 additional rows (remain unindexed)
    extra_data = generate_rows(rng, EXTRA_ROWS)
    # Offset IDs so they don't collide
    for i, row in enumerate(extra_data):
        row["id"] = SEED_ROWS + i
    table.add(extra_data)
    print(f"Appended {EXTRA_ROWS} additional rows.")

    # Capture second snapshot (DO NOT call optimize() or rebuild index)
    stats_after = table.index_stats("vector_idx")
    unindexed_after_add = int(stats_after.num_unindexed_rows)
    print(f"Second snapshot -> indexed: {int(stats_after.num_indexed_rows)}, unindexed: {unindexed_after_add}")

    # Write monitoring report
    report = {
        "index_type": index_type,
        "initial_indexed": initial_indexed,
        "initial_unindexed": initial_unindexed,
        "unindexed_after_add": unindexed_after_add,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {OUTPUT_FILE}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
