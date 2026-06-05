"""
LanceDB product-catalog search module with negation filters.

Data generation order (single RNG seeded with 2026):
  1. vectors   – shape (300, 16), float32
  2. brands    – 300 ints in [0, 10)
  3. categories – 300 ints in [0, 6)
  4. prices    – 300 floats uniform [1.0, 500.0)
  5. banned_countries – 300 ints in [0, 5)
  6. name_prefixes   – 300 ints in [0, 6)
"""

import os
import numpy as np
import pyarrow as pa
import lancedb

# ── Constants ────────────────────────────────────────────────────────────────

DB_DIR = "/home/user/myproject/lancedb"

BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE",
          "BrandF", "BrandG", "BrandH", "BrandI", "BrandJ"]

CATEGORIES = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Cat6"]

BANNED_COUNTRIES = ["", "US", "EU", "ASIA", "AFRICA"]

NAME_PREFIXES = ["premium", "budget", "classic", "deluxe", "smart", "eco"]

N_ROWS = 300
VECTOR_DIM = 16


def _table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = pa.schema([
    pa.field("id",             pa.int64()),
    pa.field("name",           pa.utf8()),
    pa.field("brand",          pa.utf8()),
    pa.field("category",       pa.utf8()),
    pa.field("price",          pa.float64()),
    pa.field("banned_country", pa.utf8()),
    pa.field("vector",         pa.list_(pa.float32(), VECTOR_DIM)),
])


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table():
    """
    (Re)create the LanceDB table from scratch, seeded deterministically.
    Returns the opened LanceDB table object.
    """
    rng = np.random.default_rng(2026)

    # 1. vectors
    vectors = rng.standard_normal((N_ROWS, VECTOR_DIM)).astype("float32")

    # 2. brands
    brand_idx = rng.integers(0, len(BRANDS), size=N_ROWS)

    # 3. categories
    cat_idx = rng.integers(0, len(CATEGORIES), size=N_ROWS)

    # 4. prices
    prices = rng.uniform(1.0, 500.0, size=N_ROWS)

    # 5. banned_countries
    bc_idx = rng.integers(0, len(BANNED_COUNTRIES), size=N_ROWS)

    # 6. name prefixes
    prefix_idx = rng.integers(0, len(NAME_PREFIXES), size=N_ROWS)

    rows = []
    for i in range(N_ROWS):
        rows.append({
            "id":             i,
            "name":           f"{NAME_PREFIXES[prefix_idx[i]]}-{i:03d}",
            "brand":          BRANDS[brand_idx[i]],
            "category":       CATEGORIES[cat_idx[i]],
            "price":          float(prices[i]),
            "banned_country": BANNED_COUNTRIES[bc_idx[i]],
            "vector":         vectors[i].tolist(),
        })

    db = lancedb.connect(DB_DIR)
    tbl = db.create_table(_table_name(), data=rows, schema=SCHEMA, mode="overwrite")
    return tbl


def _open_table():
    db = lancedb.connect(DB_DIR)
    return db.open_table(_table_name())


def _project(row: dict) -> dict:
    return {
        "id":       int(row["id"]),
        "name":     str(row["name"]),
        "brand":    str(row["brand"]),
        "category": str(row["category"]),
        "price":    float(row["price"]),
    }


# ── Search functions ──────────────────────────────────────────────────────────

def exclude_brands(query_vec: list[float], banned_brands: list[str], k: int = 10) -> list[dict]:
    """
    L2 vector search with WHERE: brand NOT IN ('<b1>','<b2>',...)
    """
    tbl = _open_table()
    quoted = ", ".join(f"'{b}'" for b in banned_brands)
    sql = f"brand NOT IN ({quoted})"
    results = tbl.search(query_vec).where(sql).limit(k).to_list()
    return [_project(r) for r in results]


def exclude_categories_and_price(query_vec: list[float], cat: str, max_price: float, k: int = 10) -> list[dict]:
    """
    L2 vector search with WHERE: category != '<cat>' AND price <= <max_price>
    """
    tbl = _open_table()
    sql = f"category != '{cat}' AND price <= {max_price}"
    results = tbl.search(query_vec).where(sql).limit(k).to_list()
    return [_project(r) for r in results]


def not_like_search(query_vec: list[float], prefix: str, k: int = 10) -> list[dict]:
    """
    L2 vector search with WHERE: name NOT LIKE '<prefix>%'
    """
    tbl = _open_table()
    sql = f"name NOT LIKE '{prefix}%'"
    results = tbl.search(query_vec).where(sql).limit(k).to_list()
    return [_project(r) for r in results]


def complex_negation(query_vec: list[float], k: int = 10) -> list[dict]:
    """
    L2 vector search with WHERE:
        NOT (brand = 'BrandA' OR (category = 'Cat3' AND price > 100.0))
    """
    tbl = _open_table()
    sql = "NOT (brand = 'BrandA' OR (category = 'Cat3' AND price > 100.0))"
    results = tbl.search(query_vec).where(sql).limit(k).to_list()
    return [_project(r) for r in results]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tbl = build_table()
    print(f"Table '{_table_name()}' created with {tbl.count_rows()} rows.")
    print(f"Schema:\n{tbl.schema}")
