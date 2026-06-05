"""
LanceDB product-catalog search module.

Table schema
------------
id              : Int64
name            : Utf8
brand           : Utf8   (BrandA … BrandJ)
category        : Utf8   (Cat1 … Cat6)
price           : Float64
banned_country  : Utf8   (one of "", "US", "EU", "ASIA", "AFRICA")
vector          : fixed_size_list<float32, 16>

RNG draw order (all from numpy.random.default_rng(2026))
---------------------------------------------------------
1. vectors      : standard_normal((300, 16))  → cast to float32
2. brand index  : integers(0, 10, size=300)
3. category idx : integers(0, 6,  size=300)
4. price        : uniform(1.0, 500.0, size=300)
5. banned idx   : integers(0, 5,  size=300)
6. name prefix  : integers(0, 6,  size=300)
"""

import os
import lancedb
import numpy as np
import pyarrow as pa


# ── constants ──────────────────────────────────────────────────────────────────
BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE",
          "BrandF", "BrandG", "BrandH", "BrandI", "BrandJ"]
CATEGORIES = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5", "Cat6"]
BANNED_COUNTRIES = ["", "US", "EU", "ASIA", "AFRICA"]
NAME_PREFIXES = ["premium", "budget", "classic", "deluxe", "smart", "eco"]

DB_PATH = "/home/user/myproject/lancedb"

SCHEMA = pa.schema([
    pa.field("id",             pa.int64()),
    pa.field("name",           pa.utf8()),
    pa.field("brand",          pa.utf8()),
    pa.field("category",       pa.utf8()),
    pa.field("price",          pa.float64()),
    pa.field("banned_country", pa.utf8()),
    pa.field("vector",         pa.list_(pa.float32(), 16)),
])


def _table_name() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    return f"products_{run_id}"


def _get_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table(_table_name())


def build_table() -> None:
    """(Re)create and populate the LanceDB products table with 300 rows."""
    rng = np.random.default_rng(2026)

    # 1. vectors
    vectors = rng.standard_normal((300, 16)).astype("float32")

    # 2. brands
    brand_idx = rng.integers(0, 10, size=300)
    brands = [BRANDS[i] for i in brand_idx]

    # 3. categories
    cat_idx = rng.integers(0, 6, size=300)
    categories = [CATEGORIES[i] for i in cat_idx]

    # 4. prices
    prices = rng.uniform(1.0, 500.0, size=300)

    # 5. banned_country
    banned_idx = rng.integers(0, 5, size=300)
    banned_countries = [BANNED_COUNTRIES[i] for i in banned_idx]

    # 6. name prefixes
    prefix_idx = rng.integers(0, 6, size=300)
    names = [f"{NAME_PREFIXES[prefix_idx[i]]}-{i:03d}" for i in range(300)]

    table = pa.table(
        {
            "id":             pa.array(list(range(300)),  type=pa.int64()),
            "name":           pa.array(names,             type=pa.utf8()),
            "brand":          pa.array(brands,            type=pa.utf8()),
            "category":       pa.array(categories,        type=pa.utf8()),
            "price":          pa.array(prices.tolist(),   type=pa.float64()),
            "banned_country": pa.array(banned_countries,  type=pa.utf8()),
            "vector":         pa.array(
                                  [v.tolist() for v in vectors],
                                  type=pa.list_(pa.float32(), 16),
                              ),
        },
        schema=SCHEMA,
    )

    db = lancedb.connect(DB_PATH)
    db.create_table(_table_name(), data=table, mode="overwrite", schema=SCHEMA)


# ── search helpers ─────────────────────────────────────────────────────────────

def _project(row: dict) -> dict:
    """Keep only the required keys, coercing types."""
    return {
        "id":       int(row["id"]),
        "name":     str(row["name"]),
        "brand":    str(row["brand"]),
        "category": str(row["category"]),
        "price":    float(row["price"]),
    }


# ── public search API ─────────────────────────────────────────────────────────

def exclude_brands(
    query_vec: list[float],
    banned_brands: list[str],
    k: int = 10,
) -> list[dict]:
    """Return up to k rows whose brand is NOT IN banned_brands."""
    tbl = _get_table()
    quoted = ", ".join(f"'{b}'" for b in banned_brands)
    sql = f"brand NOT IN ({quoted})"
    rows = tbl.search(query_vec).metric("l2").where(sql, prefilter=True).limit(k).to_list()
    return [_project(r) for r in rows]


def exclude_categories_and_price(
    query_vec: list[float],
    cat: str,
    max_price: float,
    k: int = 10,
) -> list[dict]:
    """Return up to k rows where category != cat AND price <= max_price."""
    tbl = _get_table()
    sql = f"category != '{cat}' AND price <= {max_price}"
    rows = tbl.search(query_vec).metric("l2").where(sql, prefilter=True).limit(k).to_list()
    return [_project(r) for r in rows]


def not_like_search(
    query_vec: list[float],
    prefix: str,
    k: int = 10,
) -> list[dict]:
    """Return up to k rows whose name does NOT start with prefix."""
    tbl = _get_table()
    sql = f"name NOT LIKE '{prefix}%'"
    rows = tbl.search(query_vec).metric("l2").where(sql, prefilter=True).limit(k).to_list()
    return [_project(r) for r in rows]


def complex_negation(
    query_vec: list[float],
    k: int = 10,
) -> list[dict]:
    """Return up to k rows that pass NOT (brand='BrandA' OR (category='Cat3' AND price>100.0))."""
    tbl = _get_table()
    sql = "NOT (brand = 'BrandA' OR (category = 'Cat3' AND price > 100.0))"
    rows = tbl.search(query_vec).metric("l2").where(sql, prefilter=True).limit(k).to_list()
    return [_project(r) for r in rows]


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_table()
    tbl_name = _table_name()
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(tbl_name)
    count = tbl.count_rows()
    print(f"Table '{tbl_name}' built successfully with {count} rows.")
    print(f"Schema:\n{tbl.schema}")
