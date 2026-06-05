"""
Product-catalog search module built on LanceDB with negation filters.

RNG draw order (all from numpy.random.default_rng(2026)):
  1. vectors:  rng.standard_normal((300, 16)).astype("float32")
  2. brands:   rng.choice(brand_list, size=300)
  3. categories: rng.choice(category_list, size=300)
  4. prefixes: rng.choice(prefix_list, size=300)
  5. prices:   rng.uniform(1.0, 500.0, size=300)
  6. banned_country: rng.choice(banned_country_list, size=300)
"""

import os
import numpy as np
import pyarrow as pa
import lancedb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = "/home/user/myproject/lancedb"
TABLE_NAME = f"products_{os.environ['ZEALT_RUN_ID']}"

BRANDS = [f"Brand{chr(ord('A') + i)}" for i in range(10)]          # BrandA..BrandJ
CATEGORIES = [f"Cat{i}" for i in range(1, 7)]                      # Cat1..Cat6
PREFIXES = ["premium", "budget", "classic", "deluxe", "smart", "eco"]
BANNED_COUNTRIES = ["", "US", "EU", "ASIA", "AFRICA"]

NUM_ROWS = 300
VEC_DIM = 16

# Module-level table reference (set by build_table)
_tbl = None


def _get_table():
    """Return the cached table handle, creating it if necessary."""
    global _tbl
    if _tbl is None:
        db = lancedb.connect(DB_PATH)
        _tbl = db.open_table(TABLE_NAME)
    return _tbl


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------
def build_table():
    """(Re)create the LanceDB product table with 300 deterministic rows."""
    global _tbl

    rng = np.random.default_rng(2026)

    # 1. Vectors  (must be drawn first from the RNG)
    vectors = rng.standard_normal((NUM_ROWS, VEC_DIM)).astype("float32")

    # 2. Brands
    brands = rng.choice(BRANDS, size=NUM_ROWS)

    # 3. Categories
    categories = rng.choice(CATEGORIES, size=NUM_ROWS)

    # 4. Name prefixes
    prefixes = rng.choice(PREFIXES, size=NUM_ROWS)

    # 5. Prices  (float64 in the table, drawn as float64)
    prices = rng.uniform(1.0, 500.0, size=NUM_ROWS)

    # 6. Banned countries
    banned_countries = rng.choice(BANNED_COUNTRIES, size=NUM_ROWS)

    # Build column arrays
    ids = pa.array(np.arange(NUM_ROWS, dtype=np.int64), type=pa.int64())
    names = pa.array([f"{prefixes[i]}-{i:03d}" for i in range(NUM_ROWS)], type=pa.utf8())
    brand_col = pa.array(brands.tolist(), type=pa.utf8())
    cat_col = pa.array(categories.tolist(), type=pa.utf8())
    price_col = pa.array(prices.tolist(), type=pa.float64())
    bc_col = pa.array(banned_countries.tolist(), type=pa.utf8())

    # Vector column: fixed_size_list<float32, 16>
    # Build as list of fixed-size lists
    vec_values = pa.array(vectors.flatten(), type=pa.float32())
    vec_col = pa.FixedSizeListArray.from_arrays(vec_values, list_size=VEC_DIM)

    schema = pa.schema([
        ("id", pa.int64()),
        ("name", pa.utf8()),
        ("brand", pa.utf8()),
        ("category", pa.utf8()),
        ("price", pa.float64()),
        ("banned_country", pa.utf8()),
        ("vector", pa.list_(pa.float32(), VEC_DIM)),
    ])

    table = pa.table({
        "id": ids,
        "name": names,
        "brand": brand_col,
        "category": cat_col,
        "price": price_col,
        "banned_country": bc_col,
        "vector": vec_col,
    }, schema=schema)

    db = lancedb.connect(DB_PATH)
    _tbl = db.create_table(TABLE_NAME, table, mode="overwrite")
    return _tbl


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def _search(query_vec, where_clause, k):
    """Run an L2 vector search with a where clause and return projected dicts."""
    tbl = _get_table()
    results = tbl.search(query_vec).metric("l2").where(where_clause).limit(k).to_list()
    out = []
    for row in results:
        out.append({
            "id": int(row["id"]),
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "price": float(row["price"]),
        })
    return out


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------
def exclude_brands(query_vec: list[float], banned_brands: list[str], k: int = 10) -> list[dict]:
    """Return up to *k* nearest rows whose brand is NOT IN banned_brands."""
    quoted = ",".join(f"'{b}'" for b in banned_brands)
    where_clause = f"brand NOT IN ({quoted})"
    return _search(query_vec, where_clause, k)


def exclude_categories_and_price(query_vec: list[float], cat: str, max_price: float, k: int = 10) -> list[dict]:
    """Return up to *k* nearest rows whose category != cat AND price <= max_price."""
    where_clause = f"category != '{cat}' AND price <= {max_price}"
    return _search(query_vec, where_clause, k)


def not_like_search(query_vec: list[float], prefix: str, k: int = 10) -> list[dict]:
    """Return up to *k* nearest rows whose name does NOT LIKE 'prefix%'."""
    where_clause = f"name NOT LIKE '{prefix}%'"
    return _search(query_vec, where_clause, k)


def complex_negation(query_vec: list[float], k: int = 10) -> list[dict]:
    """Return up to *k* nearest rows where NOT (brand='BrandA' OR (category='Cat3' AND price>100.0))."""
    where_clause = "NOT (brand = 'BrandA' OR (category = 'Cat3' AND price > 100.0))"
    return _search(query_vec, where_clause, k)


# ---------------------------------------------------------------------------
# Main: build the table when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tbl = build_table()
    print(f"Table '{TABLE_NAME}' created with {tbl.count_rows()} rows.")