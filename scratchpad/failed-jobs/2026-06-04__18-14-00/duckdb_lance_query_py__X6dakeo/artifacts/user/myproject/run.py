"""
DuckDB SQL Aggregation Over a LanceDB-backed Lance Dataset.

Usage:
    python3 run.py build    -- seed the LanceDB table
    python3 run.py summary  -- run DuckDB aggregations and print JSON
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
DB_DIR = "/home/user/myproject/lancedb_data"
N_ROWS = 1000
EMBEDDING_DIM = 16


def get_table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> None:
    """Create / overwrite the LanceDB products table with 1,000 deterministic rows."""
    rng = np.random.default_rng(2026)

    # Deterministic generation — exact call order matters
    category_indices = rng.integers(0, 5, size=N_ROWS)          # step 1
    prices = rng.uniform(1.0, 1000.0, size=N_ROWS)               # step 2
    in_stock_mask = rng.random(N_ROWS) < 0.7                     # step 3
    embeddings = rng.standard_normal((N_ROWS, EMBEDDING_DIM)).astype(np.float32)  # step 4

    ids = list(range(N_ROWS))
    names = [f"product_{i}" for i in range(N_ROWS)]
    categories = [CATEGORIES[idx] for idx in category_indices]

    # Build PyArrow table
    embedding_type = pa.list_(pa.float32(), EMBEDDING_DIM)

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.utf8()),
        pa.field("category", pa.utf8()),
        pa.field("price", pa.float64()),
        pa.field("in_stock", pa.bool_()),
        pa.field("embedding", embedding_type),
    ])

    arrow_table = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "name": pa.array(names, type=pa.utf8()),
            "category": pa.array(categories, type=pa.utf8()),
            "price": pa.array(prices, type=pa.float64()),
            "in_stock": pa.array(in_stock_mask.tolist(), type=pa.bool_()),
            "embedding": pa.array(
                [row.tolist() for row in embeddings],
                type=embedding_type,
            ),
        },
        schema=schema,
    )

    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)

    table_name = get_table_name()

    # Drop if exists so we always start fresh
    existing = db.table_names()
    if table_name in existing:
        db.drop_table(table_name)

    db.create_table(table_name, data=arrow_table, schema=schema)
    print(f"Built table '{table_name}' with {N_ROWS} rows in {DB_DIR}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary() -> None:
    """Open the LanceDB table, run DuckDB SQL aggregations, print JSON to stdout."""
    db = lancedb.connect(DB_DIR)
    table_name = get_table_name()

    tbl = db.open_table(table_name)

    # Obtain the underlying Lance dataset
    ds = tbl.to_lance()

    # Create an Arrow RecordBatchReader scanning only needed columns
    reader = ds.scanner(columns=["category", "price", "in_stock"]).to_reader()

    con = duckdb.connect()
    con.register("products", reader)

    rows = con.execute(
        """
        SELECT
            category,
            COUNT(*) AS n,
            AVG(price) AS avg_price,
            AVG(CAST(in_stock AS DOUBLE)) AS in_stock_rate
        FROM products
        GROUP BY category
        """
    ).fetchall()

    result = {}
    for category, n, avg_price, in_stock_rate in rows:
        result[category] = {
            "count": int(n),
            "avg_price": float(avg_price),
            "in_stock_rate": float(in_stock_rate),
        }

    # Ensure all five categories are present and in canonical order
    output = {cat: result[cat] for cat in CATEGORIES}

    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <build|summary>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "summary":
        summary()
    else:
        print(f"Unknown subcommand: {cmd!r}. Use 'build' or 'summary'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
