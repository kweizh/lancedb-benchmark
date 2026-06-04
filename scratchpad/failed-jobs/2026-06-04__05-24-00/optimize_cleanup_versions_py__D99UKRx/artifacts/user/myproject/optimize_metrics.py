"""
LanceDB compaction and old-version cleanup script.

Creates a 'metrics' table, seeds it with data, adds incremental batches,
then runs optimize with aggressive version pruning and records the state.
"""

import json
import os
import datetime
import numpy as np
import pyarrow as pa
import lancedb

URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "metrics"
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "optimize_state.json")

# --- Schema ---
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("value", pa.float32()),
    pa.field("vector", pa.list_(pa.float32(), 8)),
])


def make_rows(start_id: int, n: int, seed: int = 42) -> pa.Table:
    """Generate *n* deterministic rows beginning at *start_id*."""
    rng = np.random.default_rng(seed)
    ids = pa.array(range(start_id, start_id + n), type=pa.int64())
    values = pa.array(rng.uniform(0, 100, size=n).astype(np.float32), type=pa.float32())
    vectors = pa.array(
        rng.uniform(-1, 1, size=(n, 8)).astype(np.float32).tolist(),
        type=pa.list_(pa.float32(), 8),
    )
    return pa.table({"id": ids, "value": values, "vector": vectors}, schema=schema)


def main() -> None:
    db = lancedb.connect(URI)

    # Drop table if it already exists to ensure a clean slate
    try:
        db.drop_table(TABLE_NAME)
    except Exception:
        pass

    # Seed table with 100 rows
    initial_data = make_rows(0, 100, seed=0)
    table = db.create_table(TABLE_NAME, initial_data)

    # 8 incremental adds of 10 rows each
    next_id = 100
    for i in range(8):
        batch = make_rows(next_id, 10, seed=i + 1)
        table.add(batch)
        next_id += 10

    # Capture version count BEFORE optimize
    pre_optimize_versions = len(table.list_versions())
    print(f"Pre-optimize versions: {pre_optimize_versions}")

    # Compact fragments and prune every old version
    table.optimize(cleanup_older_than=datetime.timedelta(seconds=0))

    # Capture state AFTER optimize
    post_optimize_versions = len(table.list_versions())
    post_optimize_row_count = table.count_rows()
    print(f"Post-optimize versions: {post_optimize_versions}")
    print(f"Post-optimize row count: {post_optimize_row_count}")

    # Write result JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "pre_optimize_versions": pre_optimize_versions,
        "post_optimize_versions": post_optimize_versions,
        "post_optimize_row_count": post_optimize_row_count,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {OUTPUT_FILE}: {result}")


if __name__ == "__main__":
    main()