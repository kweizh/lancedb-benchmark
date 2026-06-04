#!/usr/bin/env python3
"""Migrate an Arrow IPC stream into LanceDB and run a top-5 nearest-neighbour search."""

import json
import os

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.ipc


def main() -> None:
    # --- Configuration ---
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"events_{run_id}"
    db_path = "/home/user/myproject/lancedb_data"
    source_path = "/app/source/dataset.arrows"
    query_path = "/app/query_vector.npy"

    # --- Read the Arrow IPC stream ---
    reader = pa.ipc.open_stream(source_path)
    source_schema = reader.schema

    # Collect all batches (the reader is single-pass)
    batches = []
    total_rows = 0
    for batch in reader:
        batches.append(batch)
        total_rows += batch.num_rows

    # --- Connect to LanceDB and create the table ---
    db = lancedb.connect(db_path)

    # Build a PyArrow Table from the collected batches to preserve schema
    table = pa.Table.from_batches(batches, schema=source_schema)

    # Drop the table if it already exists (for idempotent reruns)
    if table_name in db.table_names():
        db.drop_table(table_name)

    # Create the LanceDB table from the PyArrow Table
    lancedb_table = db.create_table(table_name, data=table)

    # --- Verify schema parity ---
    dest_schema = lancedb_table.schema
    schema_match = source_schema.equals(dest_schema, check_metadata=False)

    # --- Load query vector and run nearest-neighbour search ---
    query_vector = np.load(query_path)  # shape (48,), dtype float32

    results = (
        lancedb_table.search(query_vector.tolist())
        .limit(5)
        .to_pandas()
    )

    top5 = []
    for _, row in results.iterrows():
        top5.append({
            "id": int(row["id"]),
            "distance": float(row["_distance"]),
        })

    # --- Print result JSON to stdout ---
    output = {
        "table_name": table_name,
        "row_count": total_rows,
        "schema_match": schema_match,
        "top5": top5,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()