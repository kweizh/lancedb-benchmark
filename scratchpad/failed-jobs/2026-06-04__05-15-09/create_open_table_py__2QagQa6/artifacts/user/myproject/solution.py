"""
LanceDB Table Lifecycle: Create, Overwrite, Open, and Summarize
"""

import json
import os

import numpy as np
import pyarrow as pa
import lancedb


# ── Configuration ────────────────────────────────────────────────────────────
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_PATH = "/workspace/output/table_state.json"
TABLE_NAME  = "products"

# ── PyArrow schema ────────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("id",     pa.int32()),
    pa.field("name",   pa.string()),
    pa.field("price",  pa.float64()),
    pa.field("tags",   pa.list_(pa.string())),
    pa.field("vector", pa.list_(pa.float32(), 4)),
])


def make_batch(ids, names, prices, tags, rng: np.random.Generator) -> pa.Table:
    """Build a PyArrow table from lists + a seeded RNG for vectors."""
    n = len(ids)
    vectors = rng.random((n, 4), dtype=np.float32).tolist()
    return pa.table(
        {
            "id":     pa.array(ids,    type=pa.int32()),
            "name":   pa.array(names,  type=pa.string()),
            "price":  pa.array(prices, type=pa.float64()),
            "tags":   pa.array(tags,   type=pa.list_(pa.string())),
            "vector": pa.array(vectors, type=pa.list_(pa.float32(), 4)),
        },
        schema=SCHEMA,
    )


# ── Deterministic RNG (seed = 7) ──────────────────────────────────────────────
rng = np.random.default_rng(7)

# ── Original 6-row dataset ────────────────────────────────────────────────────
original_data = make_batch(
    ids    = [1, 2, 3, 4, 5, 6],
    names  = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Doohickey", "Thingamajig"],
    prices = [9.99, 14.99, 24.99, 34.99, 4.99, 49.99],
    tags   = [
        ["sale", "popular"],
        ["new"],
        ["electronics", "sale"],
        ["electronics"],
        ["clearance"],
        ["premium", "new"],
    ],
    rng    = rng,  # consumes the first 6 draw sets
)

# ── Interim 3-row dataset (schema-compatible, used to show overwrite) ─────────
interim_data = make_batch(
    ids    = [10, 11, 12],
    names  = ["Temp Alpha", "Temp Beta", "Temp Gamma"],
    prices = [1.00, 2.00, 3.00],
    tags   = [["temp"], ["temp"], ["temp"]],
    rng    = rng,  # continues drawing from the same RNG
)


# ── 1. Connect ────────────────────────────────────────────────────────────────
print(f"Connecting to LanceDB at: {LANCEDB_URI}")
os.makedirs(LANCEDB_URI, exist_ok=True)
db = lancedb.connect(LANCEDB_URI)

# ── 2. Initial create ─────────────────────────────────────────────────────────
print("Creating 'products' table with original 6 rows …")
db.create_table(TABLE_NAME, data=original_data, schema=SCHEMA, mode="overwrite")

# ── 3. Demonstrate overwrite with different rows ──────────────────────────────
print("Overwriting 'products' with interim 3-row dataset …")
db.create_table(TABLE_NAME, data=interim_data, schema=SCHEMA, mode="overwrite")
interim_count = db.open_table(TABLE_NAME).count_rows()
print(f"  Row count after interim overwrite: {interim_count}")  # should be 3

# ── 4. Restore original 6-row dataset via overwrite ──────────────────────────
print("Restoring original 6-row dataset via overwrite …")
# Re-seed the RNG to reproduce identical vectors for the original rows
rng_restore = np.random.default_rng(7)
original_data_restored = make_batch(
    ids    = [1, 2, 3, 4, 5, 6],
    names  = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Doohickey", "Thingamajig"],
    prices = [9.99, 14.99, 24.99, 34.99, 4.99, 49.99],
    tags   = [
        ["sale", "popular"],
        ["new"],
        ["electronics", "sale"],
        ["electronics"],
        ["clearance"],
        ["premium", "new"],
    ],
    rng    = rng_restore,
)
db.create_table(TABLE_NAME, data=original_data_restored, schema=SCHEMA, mode="overwrite")

# ── 5. Reopen the table and inspect ──────────────────────────────────────────
print("Reopening table with db.open_table() …")
table = db.open_table(TABLE_NAME)
row_count = table.count_rows()
print(f"  Final row count : {row_count}")

arrow_schema        = table.schema
schema_field_names  = sorted(arrow_schema.names)
tables_in_db        = sorted(db.table_names())

print(f"  Tables in DB    : {tables_in_db}")
print(f"  Schema fields   : {schema_field_names}")

# ── 6. Write JSON summary ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
summary = {
    "tables_in_db":       tables_in_db,
    "row_count":          row_count,
    "schema_field_names": schema_field_names,
}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nJSON summary written to {OUTPUT_PATH}")
print(json.dumps(summary, indent=2))
