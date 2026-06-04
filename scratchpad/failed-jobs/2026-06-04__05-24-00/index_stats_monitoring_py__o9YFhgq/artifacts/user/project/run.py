"""Monitor LanceDB IVF_PQ index health using index_stats."""

import json
import os
import datetime

import numpy as np
import pyarrow as pa
import lancedb


def main():
    # 1. Connect to LanceDB
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)

    # 2. Drop pre-existing table to keep the run idempotent, then create fresh
    if "points" in [t.name for t in db.table_names()]:
        db.drop_table("points")

    # Define schema: id (int64) + vector (fixed_size_list<float32>[16])
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 16)),
    ])

    # 3. Seed with exactly 400 deterministic rows from numpy.random.default_rng(50)
    rng = np.random.default_rng(50)
    ids = pa.array(range(400), type=pa.int64())
    vectors = pa.array(rng.random((400, 16)).astype(np.float32).tolist(), type=pa.list_(pa.float32(), 16))
    seed_table = pa.table({"id": ids, "vector": vectors})

    table = db.create_table("points", data=seed_table, schema=schema)

    # 4. Build IVF_PQ index over the vector column
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        replace=True,
    )
    table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=120))

    # 5. Capture initial snapshot of index stats
    stats_initial = table.index_stats("vector_idx")
    initial_indexed = int(stats_initial.num_indexed_rows)
    initial_unindexed = int(stats_initial.num_unindexed_rows)
    index_type = stats_initial.index_type

    print(f"Initial stats: indexed={initial_indexed}, unindexed={initial_unindexed}, type={index_type}")

    # 6. Append 50 additional deterministic rows (DO NOT optimize/rebuild)
    rng2 = np.random.default_rng(51)
    new_ids = pa.array(range(400, 450), type=pa.int64())
    new_vectors = pa.array(rng2.random((50, 16)).astype(np.float32).tolist(), type=pa.list_(pa.float32(), 16))
    new_rows = pa.table({"id": new_ids, "vector": new_vectors})
    table.add(new_rows)

    # 7. Capture second snapshot of index stats
    stats_after = table.index_stats("vector_idx")
    unindexed_after_add = int(stats_after.num_unindexed_rows)

    print(f"After add stats: unindexed={unindexed_after_add}")

    # Write monitoring report
    os.makedirs("/workspace/output", exist_ok=True)
    report = {
        "index_type": index_type,
        "initial_indexed": initial_indexed,
        "initial_unindexed": initial_unindexed,
        "unindexed_after_add": unindexed_after_add,
    }

    with open("/workspace/output/index_stats.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to /workspace/output/index_stats.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()