import sys
import os
import json
import numpy as np
import pyarrow as pa
import lancedb
import duckdb

def get_db_and_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    table_name = f"products_{run_id}"
    db = lancedb.connect("lancedb_data")
    return db, table_name

def build():
    db, table_name = get_db_and_table_name()
    
    # Deterministic generation
    rng = np.random.default_rng(2026)
    category_indices = rng.integers(0, 5, size=1000)
    prices = rng.uniform(1.0, 1000.0, size=1000)
    in_stock_mask = rng.random(1000) < 0.7
    embeddings = rng.standard_normal((1000, 16)).astype(np.float32)
    
    categories = ["books", "electronics", "clothing", "food", "toys"]
    category_names = [categories[idx] for idx in category_indices]
    
    ids = list(range(1000))
    names = [f"product_{i}" for i in range(1000)]
    
    schema = pa.schema([
        ("id", pa.int64()),
        ("name", pa.string()),
        ("category", pa.string()),
        ("price", pa.float64()),
        ("in_stock", pa.bool_()),
        ("embedding", pa.list_(pa.float32(), 16))
    ])
    
    table = pa.Table.from_pydict({
        "id": ids,
        "name": names,
        "category": category_names,
        "price": prices,
        "in_stock": in_stock_mask,
        "embedding": list(embeddings)
    }, schema=schema)
    
    # Create / overwrite the table
    db.create_table(table_name, data=table, mode="overwrite")
    sys.exit(0)

def summary():
    db, table_name = get_db_and_table_name()
    tbl = db.open_table(table_name)
    ds = tbl.to_lance()
    
    # Use DuckDB to run SQL aggregations over the Lance dataset
    reader = ds.scanner(columns=["category", "price", "in_stock"]).to_reader()
    con = duckdb.connect()
    con.register("products", reader)
    
    rows = con.execute("""
        SELECT 
            category, 
            COUNT(*) AS n, 
            AVG(price) AS avg_price, 
            AVG(CAST(in_stock AS DOUBLE)) AS in_stock_rate 
        FROM products 
        GROUP BY category
    """).fetchall()
    
    res = {
        "books": {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "electronics": {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "clothing": {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "food": {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "toys": {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0}
    }
    
    for category, count, avg_price, in_stock_rate in rows:
        if category in res:
            res[category] = {
                "count": int(count),
                "avg_price": float(avg_price) if avg_price is not None else 0.0,
                "in_stock_rate": float(in_stock_rate) if in_stock_rate is not None else 0.0
            }
            
    print(json.dumps(res))
    sys.exit(0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run.py [build|summary]", file=sys.stderr)
        sys.exit(1)
        
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "summary":
        summary()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
