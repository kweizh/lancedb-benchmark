#!/usr/bin/env python3
"""
LanceDB + DuckDB product aggregation demo.

Subcommands:
  build    — creates / overwrites the products_${ZEALT_RUN_ID} table (1 000 rows)
  summary  — runs DuckDB SQL over the Lance dataset and prints JSON stats
"""

import json
import os
import sys

import duckdb
import lancedb
import numpy as np
import pyarrow as pa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["books", "electronics", "clothing", "food", "toys"]
NUM_ROWS = 1_000
DB_DIR = "./lancedb_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"


def _build_arrow_table() -> pa.Table:
    """Generate the deterministic 1 000-row product table as a PyArrow Table."""
    rng = np.random.default_rng(2026)

    # --- deterministic generation order (must match spec exactly) ---
    category_indices = rng.integers(0, 5, size=NUM_ROWS)          # step 1
    prices = rng.uniform(1.0, 1000.0, size=NUM_ROWS)              # step 2
    in_stock_mask = rng.random(NUM_ROWS) < 0.7                    # step 3
    embeddings = rng.standard_normal((NUM_ROWS, 16)).astype(np.float32)  # step 4

    ids = list(range(NUM_ROWS))
    names = [f"product_{i}" for i in range(NUM_ROWS)]
    categories = [CATEGORIES[idx] for idx in category_indices]

    # Build PyArrow arrays
    embedding_type = pa.list_(pa.float32(), 16)  # fixed-size list of float32

    arrow_table = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "name": pa.array(names, type=pa.utf8()),
            "category": pa.array(categories, type=pa.utf8()),
            "price": pa.array(prices.tolist(), type=pa.float64()),
            "in_stock": pa.array(in_stock_mask.tolist(), type=pa.bool_()),
            "embedding": pa.array(
                [row.tolist() for row in embeddings],
                type=embedding_type,
            ),
        }
    )
    return arrow_table


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_build() -> None:
    """Create / overwrite the LanceDB products table."""
    tbl_name = _table_name()
    db = lancedb.connect(DB_DIR)

    # Drop existing table if present so we can recreate deterministically
    existing = db.table_names()
    if tbl_name in existing:
        db.drop_table(tbl_name)

    arrow_table = _build_arrow_table()
    tbl = db.create_table(tbl_name, data=arrow_table)

    print(f"Created table '{tbl_name}' with {tbl.count_rows()} rows.")


def cmd_summary() -> None:
    """Run DuckDB SQL aggregations over the Lance dataset and print JSON."""
    tbl_name = _table_name()
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(tbl_name)

    # Expose the underlying Lance dataset and create an Arrow RecordBatchReader
    ds = tbl.to_lance()
    reader = ds.scanner(columns=["category", "price", "in_stock"]).to_reader()

    # Register with DuckDB and run aggregation SQL
    con = duckdb.connect()
    con.register("products", reader)
    rows = con.execute(
        """
        SELECT
            category,
            COUNT(*)                                   AS n,
            AVG(price)                                 AS avg_price,
            AVG(CAST(in_stock AS DOUBLE))              AS in_stock_rate
        FROM products
        GROUP BY category
        """
    ).fetchall()

    # Build result dict in the required shape
    result: dict = {}
    for category, n, avg_price, in_stock_rate in rows:
        result[category] = {
            "count": int(n),
            "avg_price": float(avg_price),
            "in_stock_rate": float(in_stock_rate),
        }

    # Ensure all five categories are present (even if count were 0)
    for cat in CATEGORIES:
        if cat not in result:
            result[cat] = {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0}

    # Print in canonical category order
    ordered = {cat: result[cat] for cat in CATEGORIES}
    print(json.dumps(ordered, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <build|summary>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        cmd_build()
    elif cmd == "summary":
        cmd_summary()
    else:
        print(f"Unknown subcommand: {cmd!r}. Use 'build' or 'summary'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
