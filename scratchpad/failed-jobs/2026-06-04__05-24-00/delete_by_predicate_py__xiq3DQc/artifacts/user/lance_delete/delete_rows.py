#!/usr/bin/env python3
"""Seed a LanceDB logs table, delete rows by SQL predicates, and export surviving state."""

import json
import os

import lancedb
import pyarrow as pa

# ── Configuration ──────────────────────────────────────────────────────────
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "logs"
OUTPUT_PATH = "/workspace/output/delete_state.json"

# ── Schema definition (column order matters) ──────────────────────────────
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("level", pa.string()),
    pa.field("seq", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), list_size=4)),
])

# ── Seed data ──────────────────────────────────────────────────────────────
levels = ["info", "warn", "error"]
ids = list(range(1, 51))  # 1..=50

id_col = pa.array(ids, type=pa.int64())
level_col = pa.array([levels[(i - 1) % 3] for i in ids], type=pa.string())
seq_col = pa.array([i * 2 for i in ids], type=pa.int32())

# Deterministic vectors: each is [id*0.1, id*0.2, id*0.3, id*0.4] as float32
vector_col = pa.array(
    [[float(i * 0.1), float(i * 0.2), float(i * 0.3), float(i * 0.4)] for i in ids],
    type=pa.list_(pa.float32(), list_size=4),
)

seed_table = pa.table(
    [id_col, level_col, seq_col, vector_col],
    schema=schema,
)

# ── Connect & create table ────────────────────────────────────────────────
db = lancedb.connect(LANCEDB_URI)

# Drop existing table if present so we start fresh
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)

tbl = db.create_table(TABLE_NAME, seed_table)

print(f"Seeded {tbl.count_rows()} rows into '{TABLE_NAME}'")

# ── Deletes (in order) ────────────────────────────────────────────────────
# 1. Delete every row whose level equals 'warn'
tbl.delete("level = 'warn'")
print(f"After deleting warn: {tbl.count_rows()} rows")

# 2. Delete every row whose level equals 'info' AND seq > 60
tbl.delete("level = 'info' AND seq > 60")
print(f"After deleting info with seq > 60: {tbl.count_rows()} rows")

# 3. Delete every row whose id is in {5, 9, 13}
tbl.delete("id IN (5, 9, 13)")
print(f"After deleting ids 5, 9, 13: {tbl.count_rows()} rows")

# ── Export result ──────────────────────────────────────────────────────────
total_rows = tbl.count_rows()
remaining_ids = sorted(tbl.to_pandas()["id"].tolist())

output = {
    "total_rows": total_rows,
    "remaining_ids_sorted": remaining_ids,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {OUTPUT_PATH}: {json.dumps(output)}")