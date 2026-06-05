#!/usr/bin/env python3
"""CLI tool to build and summarize a LanceDB product table using DuckDB SQL aggregations."""

import json
import os
import sys

import duckdb
import lancedb
import numpy as np
import pyarrow as pa

CATEGORIES = ["books", "electronics", "clothing", "food", "toys"]
DB_PATH = "/home/user/myproject/lancedb_data"


def get_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    return f"products_{run_id}"


def build():
    """Create/overwrite the LanceDB table with 1,000 deterministic product rows."""
    rng = np.random.default_rng(2026)

    # Deterministic generation order per spec
    category_indices = rng.integers(0, 5, size=1000)
    prices = rng.uniform(1.0, 1000.0, size=1000)
    in_stock_mask = rng.random(1000) < 0.7
    embeddings = rng.standard_normal((1000, 16)).astype(np.float32)

    # Build arrays
    ids = pa.array(range(1000), type=pa.int64())
    names = pa.array([f"product_{i}" for i in range(1000)], type=pa.utf8())
    categories = pa.array(
        [CATEGORIES[idx] for idx in category_indices], type=pa.utf8()
    )
    price_arr = pa.array(prices, type=pa.float64())
    in_stock_arr = pa.array(in_stock_mask, type=pa.bool_())

    # Build fixed-size-list for embeddings (16-dimensional Float32 vectors)
    flat_embeddings = embeddings.flatten().tolist()
    embedding_values = pa.array(flat_embeddings, type=pa.float32())
    embedding_arr = pa.FixedSizeListArray.from_arrays(
        embedding_values, list_size=16
    )

    table = pa.table(
        {
            "id": ids,
            "name": names,
            "category": categories,
            "price": price_arr,
            "in_stock": in_stock_arr,
            "embedding": embedding_arr,
        }
    )

    table_name = get_table_name()
    db = lancedb.connect(DB_PATH)

    # Drop existing table if it exists, then create fresh
    try:
        db.drop_table(table_name)
    except Exception:
        pass

    db.create_table(table_name, table)
    print(f"Created table '{table_name}' with 1,000 rows.", file=sys.stderr)
    sys.exit(0)


def summary():
    """Open the LanceDB table, run DuckDB SQL aggregations, and print JSON summary."""
    table_name = get_table_name()
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(table_name)

    # Get the underlying Lance dataset
    ds = tbl.to_lance()

    # Create an Arrow scanner over the Lance dataset with only needed columns
    reader = ds.scanner(columns=["category", "price", "in_stock"]).to_reader()

    # Use DuckDB to run SQL aggregations
    con = duckdb.connect()
    con.register("products", reader)

    rows = con.execute(
        "SELECT category, COUNT(*) AS n, AVG(price) AS avg_price, "
        "AVG(CAST(in_stock AS DOUBLE)) AS in_stock_rate "
        "FROM products GROUP BY category"
    ).fetchall()

    # Build the result dict with all five categories initialized
    result = {
        cat: {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0}
        for cat in CATEGORIES
    }

    for category, count, avg_price, in_stock_rate in rows:
        result[category] = {
            "count": int(count),
            "avg_price": float(avg_price),
            "in_stock_rate": float(in_stock_rate),
        }

    print(json.dumps(result))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "summary"):
        print("Usage: python3 run.py {build|summary}", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    if command == "build":
        build()
    elif command == "summary":
        summary()


if __name__ == "__main__":
    main()