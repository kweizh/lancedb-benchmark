#!/usr/bin/env python3
"""
Migrate an Arrow IPC stream into LanceDB and run a top-5 nearest-neighbour search.
"""
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import lancedb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_PATH = "/app/source/dataset.arrows"
QUERY_VECTOR_PATH = "/app/query_vector.npy"
DB_PATH = "/home/user/myproject/lancedb_data"
VECTOR_COLUMN = "embedding"

run_id = os.environ["ZEALT_RUN_ID"]
TABLE_NAME = f"events_{run_id}"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Read the Arrow IPC stream
    # ------------------------------------------------------------------
    with open(SOURCE_PATH, "rb") as f:
        reader = ipc.open_stream(f)
        source_schema: pa.Schema = reader.schema
        source_table: pa.Table = reader.read_all()

    row_count: int = source_table.num_rows

    # ------------------------------------------------------------------
    # 2. Connect to LanceDB and (re-)create the table
    # ------------------------------------------------------------------
    os.makedirs(DB_PATH, exist_ok=True)

    db = lancedb.connect(DB_PATH)

    # Drop and recreate if the table already exists so reruns are idempotent
    existing = list(db.table_names())
    if TABLE_NAME in existing:
        db.drop_table(TABLE_NAME)

    # Pass the RecordBatchStreamReader directly to preserve the exact schema.
    # RecordBatchStreamReader implements __arrow_c_stream__ so LanceDB accepts it
    # without any pandas round-trip that would silently widen fixed_size_list→list.
    with open(SOURCE_PATH, "rb") as f:
        stream_reader = ipc.open_stream(f)
        table = db.create_table(TABLE_NAME, data=stream_reader)

    # ------------------------------------------------------------------
    # 3. Verify schema parity
    # ------------------------------------------------------------------
    dest_schema: pa.Schema = table.schema

    # LanceDB may append internal metadata fields (e.g. _rowid, _rowaddr);
    # compare only the source fields, ignoring extra metadata bytes.
    source_fields = {f.name: f for f in source_schema}
    dest_fields   = {f.name: f for f in dest_schema}

    schema_match: bool = all(
        name in dest_fields and dest_fields[name].type == field.type
        for name, field in source_fields.items()
    )

    # ------------------------------------------------------------------
    # 4. Load the query vector and run top-5 ANN search
    # ------------------------------------------------------------------
    query_vector: np.ndarray = np.load(QUERY_VECTOR_PATH).astype(np.float32)

    results = (
        table.search(query_vector, vector_column_name=VECTOR_COLUMN)
        .metric("l2")
        .limit(5)
        .select(["id", "_distance"])
        .to_arrow()
    )

    top5 = [
        {
            "id": int(results["id"][i].as_py()),
            "distance": float(results["_distance"][i].as_py()),
        }
        for i in range(len(results))
    ]

    # ------------------------------------------------------------------
    # 5. Emit the JSON result
    # ------------------------------------------------------------------
    output = {
        "table_name": TABLE_NAME,
        "row_count": row_count,
        "schema_match": schema_match,
        "top5": top5,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
