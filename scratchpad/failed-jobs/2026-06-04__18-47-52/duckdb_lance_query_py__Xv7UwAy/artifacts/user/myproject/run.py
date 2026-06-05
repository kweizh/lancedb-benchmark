import os
import sys
import json
import argparse
import numpy as np
import pyarrow as pa
import lancedb
import duckdb
from lancedb.pydantic import LanceModel, Vector

def get_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"

def build():
    db_path = "./data.lancedb"
    db = lancedb.connect(db_path)
    
    rng = np.random.default_rng(2026)
    category_indices = rng.integers(0, 5, size=1000)
    prices = rng.uniform(1.0, 1000.0, size=1000)
    in_stock_mask = rng.random(1000) < 0.7
    embeddings = rng.standard_normal((1000, 16)).astype(np.float32)
    
    categories = ["books", "electronics", "clothing", "food", "toys"]
    
    data = []
    for i in range(1000):
        data.append({
            "id": i,
            "name": f"product_{i}",
            "category": categories[category_indices[i]],
            "price": float(prices[i]),
            "in_stock": bool(in_stock_mask[i]),
            "embedding": embeddings[i]
        })
    
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("category", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("in_stock", pa.bool_()),
        pa.field("embedding", pa.list_(pa.float32(), 16))
    ])
    
    table_name = get_table_name()
    db.create_table(table_name, data=data, schema=schema, mode="overwrite")

def summary():
    db_path = "./data.lancedb"
    db = lancedb.connect(db_path)
    
    table_name = get_table_name()
    tbl = db.open_table(table_name)
    
    ds = tbl.to_lance()
    reader = ds.scanner(columns=["category", "price", "in_stock"]).to_reader()
    
    con = duckdb.connect()
    con.register("products", reader)
    
    query = """
    SELECT 
        category, 
        COUNT(*) AS n, 
        AVG(price) AS avg_price, 
        AVG(CAST(in_stock AS DOUBLE)) AS in_stock_rate 
    FROM products 
    GROUP BY category
    """
    
    rows = con.execute(query).fetchall()
    
    result = {
        "books":       {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "electronics": {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "clothing":    {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "food":        {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0},
        "toys":        {"count": 0, "avg_price": 0.0, "in_stock_rate": 0.0}
    }
    
    for row in rows:
        cat = row[0]
        n = row[1]
        avg_price = row[2]
        in_stock_rate = row[3]
        if cat in result:
            result[cat] = {
                "count": int(n),
                "avg_price": float(avg_price),
                "in_stock_rate": float(in_stock_rate)
            }
            
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "summary"])
    args = parser.parse_args()
    
    if args.command == "build":
        build()
    elif args.command == "summary":
        summary()
