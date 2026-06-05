import os
import lancedb
import pyarrow as pa
import numpy as np

DB_PATH = "/home/user/myproject/lancedb"

def get_table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"

def build_table():
    db = lancedb.connect(DB_PATH)
    table_name = get_table_name()
    
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.utf8()),
        pa.field("brand", pa.utf8()),
        pa.field("category", pa.utf8()),
        pa.field("price", pa.float64()),
        pa.field("banned_country", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), 16))
    ])
    
    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((300, 16)).astype("float32")
    
    brands_list = [f"Brand{chr(65+i)}" for i in range(10)]
    brands = rng.choice(brands_list, size=300)
    
    categories_list = [f"Cat{i+1}" for i in range(6)]
    categories = rng.choice(categories_list, size=300)
    
    countries_list = ["", "US", "EU", "ASIA", "AFRICA"]
    banned_countries = rng.choice(countries_list, size=300)
    
    prefixes_list = ["premium", "budget", "classic", "deluxe", "smart", "eco"]
    prefixes = rng.choice(prefixes_list, size=300)
    
    prices = rng.uniform(10.0, 1000.0, size=300)
    
    data = []
    for i in range(300):
        data.append({
            "id": i,
            "name": f"{prefixes[i]}-{i:03d}",
            "brand": brands[i],
            "category": categories[i],
            "price": float(prices[i]),
            "banned_country": banned_countries[i],
            "vector": vectors[i].tolist()
        })
    
    db.create_table(table_name, schema=schema, data=data, mode="overwrite")

def _get_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table(get_table_name())

def _format_result(results):
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "brand": r["brand"],
            "category": r["category"],
            "price": r["price"]
        }
        for r in results
    ]

def exclude_brands(query_vec: list[float], banned_brands: list[str], k: int = 10) -> list[dict]:
    tbl = _get_table()
    if not banned_brands:
        banned_str = "''"
    else:
        banned_str = ",".join([f"'{b}'" for b in banned_brands])
    
    where_clause = f"brand NOT IN ({banned_str})"
    res = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _format_result(res)

def exclude_categories_and_price(query_vec: list[float], cat: str, max_price: float, k: int = 10) -> list[dict]:
    tbl = _get_table()
    where_clause = f"category != '{cat}' AND price <= {max_price}"
    res = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _format_result(res)

def not_like_search(query_vec: list[float], prefix: str, k: int = 10) -> list[dict]:
    tbl = _get_table()
    where_clause = f"name NOT LIKE '{prefix}%'"
    res = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _format_result(res)

def complex_negation(query_vec: list[float], k: int = 10) -> list[dict]:
    tbl = _get_table()
    where_clause = "NOT (brand = 'BrandA' OR (category = 'Cat3' AND price > 100.0))"
    res = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _format_result(res)

if __name__ == "__main__":
    build_table()
