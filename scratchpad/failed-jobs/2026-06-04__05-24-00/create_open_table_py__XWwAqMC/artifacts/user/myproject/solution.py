#!/usr/bin/env python3
"""LanceDB Table Lifecycle: Create, Overwrite, Open, and Summarize."""

import json
import os

import lancedb
import numpy as np
import pyarrow as pa


def main():
    # --- Configuration ---
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")

    # --- Connect to LanceDB ---
    db = lancedb.connect(db_uri)

    # --- Define the explicit PyArrow schema ---
    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("vector", pa.list_(pa.float32(), 4)),  # fixed_size_list<float32>[4]
    ])

    # --- Generate deterministic vector data ---
    rng = np.random.default_rng(7)

    # --- Seed the table with 6 rows of deterministic data ---
    original_data = pa.table({
        "id": pa.array([1, 2, 3, 4, 5, 6], type=pa.int32()),
        "name": pa.array([
            "Widget A",
            "Widget B",
            "Gadget C",
            "Gadget D",
            "Doohickey E",
            "Doohickey F",
        ], type=pa.string()),
        "price": pa.array([9.99, 19.99, 29.99, 39.99, 49.99, 59.99], type=pa.float64()),
        "tags": pa.array([
            ["sale", "popular"],
            ["new"],
            ["sale", "clearance"],
            ["featured"],
            ["new", "popular"],
            ["clearance"],
        ], type=pa.list_(pa.string())),
        "vector": pa.array(
            rng.random((6, 4)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32(), 4),
        ),
    }, schema=schema)

    # --- Create the products table ---
    db.create_table("products", original_data, schema=schema, mode="create")

    # --- Demonstrate mode="overwrite" with a different (but schema-compatible) set of rows ---
    overwrite_data = pa.table({
        "id": pa.array([10, 11], type=pa.int32()),
        "name": pa.array(["Overwrite Item 1", "Overwrite Item 2"], type=pa.string()),
        "price": pa.array([1.99, 2.99], type=pa.float64()),
        "tags": pa.array([["temp"], ["temp"]], type=pa.list_(pa.string())),
        "vector": pa.array(
            rng.random((2, 4)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32(), 4),
        ),
    }, schema=schema)

    db.create_table("products", overwrite_data, schema=schema, mode="overwrite")

    # --- Restore the original 6-row dataset via mode="overwrite" ---
    db.create_table("products", original_data, schema=schema, mode="overwrite")

    # --- Reopen the table and verify ---
    table = db.open_table("products")
    row_count = table.count_rows()

    # --- Build the JSON summary ---
    table_names = sorted(db.table_names())
    schema_field_names = sorted([field.name for field in table.schema])

    summary = {
        "tables_in_db": table_names,
        "row_count": row_count,
        "schema_field_names": schema_field_names,
    }

    # --- Write the JSON summary ---
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "table_state.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()