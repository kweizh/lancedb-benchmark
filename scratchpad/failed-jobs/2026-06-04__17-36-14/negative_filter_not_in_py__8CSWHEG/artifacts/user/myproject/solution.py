import os
import numpy as np
import pyarrow as pa
import lancedb

DB_PATH = "/home/user/myproject/lancedb"

def _get_table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"

def build_table():
    """
    Deterministically creates a LanceDB table named products_${ZEALT_RUN_ID}
    and populates it with exactly 300 rows.
    """
    # Seed every random value with numpy.random.default_rng(2026)
    rng = np.random.default_rng(2026)
    
    # 1. Generate vectors first
    vectors = rng.standard_normal((300, 16)).astype("float32")
    
    prefixes = ["premium", "budget", "classic", "deluxe", "smart", "eco"]
    brands = [f"Brand{chr(65 + i)}" for i in range(10)]  # BrandA .. BrandJ
    categories = [f"Cat{i}" for i in range(1, 7)]       # Cat1 .. Cat6
    banned_countries = ["", "US", "EU", "ASIA", "AFRICA"]
    
    ids = []
    names = []
    brands_col = []
    categories_col = []
    prices = []
    banned_countries_col = []
    vectors_list = []
    
    # 2. Generate other attributes row-by-row in a fixed, documented order
    for i in range(300):
        prefix = rng.choice(prefixes)
        name = f"{prefix}-{i:03d}"
        brand = rng.choice(brands)
        category = rng.choice(categories)
        price = float(rng.uniform(10.0, 200.0))
        banned_country = rng.choice(banned_countries)
        
        ids.append(i)
        names.append(name)
        brands_col.append(brand)
        categories_col.append(category)
        prices.append(price)
        banned_countries_col.append(banned_country)
        vectors_list.append(list(vectors[i]))
        
    # Define PyArrow schema explicitly
    schema = pa.schema([
        ('id', pa.int64()),
        ('name', pa.string()),
        ('brand', pa.string()),
        ('category', pa.string()),
        ('price', pa.float64()),
        ('banned_country', pa.string()),
        ('vector', pa.list_(pa.float32(), 16))
    ])
    
    # Create PyArrow Table
    arrays = [
        pa.array(ids, type=pa.int64()),
        pa.array(names, type=pa.string()),
        pa.array(brands_col, type=pa.string()),
        pa.array(categories_col, type=pa.string()),
        pa.array(prices, type=pa.float64()),
        pa.array(banned_countries_col, type=pa.string()),
        pa.array(vectors_list, type=pa.list_(pa.float32(), 16))
    ]
    pa_table = pa.Table.from_arrays(arrays, schema=schema)
    
    # Connect and create table
    os.makedirs(DB_PATH, exist_ok=True)
    db = lancedb.connect(DB_PATH)
    table_name = _get_table_name()
    db.create_table(table_name, data=pa_table, schema=schema, mode="overwrite")

def _get_table():
    db = lancedb.connect(DB_PATH)
    table_name = _get_table_name()
    return db.open_table(table_name)

def _project_results(rows) -> list[dict]:
    results = []
    for row in rows:
        results.append({
            "id": int(row["id"]),
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "price": float(row["price"])
        })
    return results

def exclude_brands(query_vec: list[float], banned_brands: list[str], k: int = 10) -> list[dict]:
    tbl = _get_table()
    if not banned_brands:
        where_clause = "brand NOT IN ('')"
    else:
        escaped_brands = [b.replace("'", "''") for b in banned_brands]
        brands_str = ", ".join(f"'{b}'" for b in escaped_brands)
        where_clause = f"brand NOT IN ({brands_str})"
    
    rows = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _project_results(rows)

def exclude_categories_and_price(query_vec: list[float], cat: str, max_price: float, k: int = 10) -> list[dict]:
    tbl = _get_table()
    escaped_cat = cat.replace("'", "''")
    where_clause = f"category != '{escaped_cat}' AND price <= {max_price}"
    
    rows = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _project_results(rows)

def not_like_search(query_vec: list[float], prefix: str, k: int = 10) -> list[dict]:
    tbl = _get_table()
    escaped_prefix = prefix.replace("'", "''")
    where_clause = f"name NOT LIKE '{escaped_prefix}%'"
    
    rows = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _project_results(rows)

def complex_negation(query_vec: list[float], k: int = 10) -> list[dict]:
    tbl = _get_table()
    where_clause = "NOT (brand = 'BrandA' OR (category = 'Cat3' AND price > 100.0))"
    
    rows = tbl.search(query_vec).where(where_clause).limit(k).to_list()
    return _project_results(rows)

if __name__ == "__main__":
    build_table()
