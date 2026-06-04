"""
LanceDB Compaction and Old-Version Cleanup Script

Connects to a LanceDB store, creates a 'metrics' table, performs multiple
small writes to accumulate fragments/versions, then runs optimize/cleanup
and writes the result to /workspace/output/optimize_state.json.
"""

import json
import os
from datetime import timedelta

import numpy as np
import pyarrow as pa
import lancedb


def make_batch(n: int, id_offset: int = 0) -> pa.Table:
    """Generate a PyArrow table with `n` rows of deterministic data."""
    rng = np.random.default_rng(seed=id_offset)
    ids = pa.array(range(id_offset, id_offset + n), type=pa.int64())
    values = pa.array(rng.random(n, dtype=np.float32), type=pa.float32())
    # 8-dimensional float32 vectors
    raw_vectors = rng.random((n, 8), dtype=np.float32)
    vector_type = pa.list_(pa.float32(), 8)
    vectors = pa.array(
        [row.tolist() for row in raw_vectors],
        type=vector_type,
    )
    return pa.table({"id": ids, "value": values, "vector": vectors})


def main() -> None:
    # --- Connection ----------------------------------------------------------
    uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(uri)

    # --- Schema --------------------------------------------------------------
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("value", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), 8)),
        ]
    )

    # --- Create table with 100 seed rows -------------------------------------
    table_name = "metrics"
    # Drop if already exists so the script is idempotent
    if table_name in db.table_names():
        db.drop_table(table_name)

    seed_batch = make_batch(100, id_offset=0)
    table = db.create_table(table_name, data=seed_batch, schema=schema)

    # --- 8 small add() calls of 10 rows each ---------------------------------
    for i in range(8):
        batch = make_batch(10, id_offset=100 + i * 10)
        table.add(batch)

    # --- Capture version count BEFORE optimize -------------------------------
    pre_optimize_versions = len(table.list_versions())

    # --- Compact fragments and prune all old versions ------------------------
    table.optimize(cleanup_older_than=timedelta(seconds=0))

    # --- Capture metrics AFTER optimize --------------------------------------
    post_optimize_versions = len(table.list_versions())
    post_optimize_row_count = table.count_rows()

    # --- Write result JSON ---------------------------------------------------
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "optimize_state.json")

    result = {
        "pre_optimize_versions": pre_optimize_versions,
        "post_optimize_versions": post_optimize_versions,
        "post_optimize_row_count": post_optimize_row_count,
    }

    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"pre_optimize_versions  : {pre_optimize_versions}")
    print(f"post_optimize_versions : {post_optimize_versions}")
    print(f"post_optimize_row_count: {post_optimize_row_count}")
    print(f"Result written to      : {output_path}")

    # --- Validate acceptance criteria ----------------------------------------
    assert post_optimize_row_count == 180, (
        f"Expected 180 rows, got {post_optimize_row_count}"
    )
    assert pre_optimize_versions > post_optimize_versions, (
        f"Expected pre ({pre_optimize_versions}) > post ({post_optimize_versions})"
    )
    assert post_optimize_versions >= 1, (
        f"Expected at least 1 version, got {post_optimize_versions}"
    )
    print("All acceptance criteria passed.")


if __name__ == "__main__":
    main()
