"""
LanceDB log-table cleanup script.

1. Connects to LanceDB at LANCEDB_URI (default: /workspace/db).
2. Creates (or recreates) a 'logs' table with schema: id, level, seq, vector.
3. Seeds 50 deterministic rows.
4. Runs three SQL-predicate deletes in order.
5. Writes /workspace/output/delete_state.json with total_rows and remaining_ids_sorted.
"""

import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import lancedb

# ── Configuration ─────────────────────────────────────────────────────────────
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_PATH = pathlib.Path("/workspace/output/delete_state.json")

# ── Schema ────────────────────────────────────────────────────────────────────
VECTOR_DIM = 4
SCHEMA = pa.schema([
    pa.field("id",     pa.int64()),
    pa.field("level",  pa.string()),
    pa.field("seq",    pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
])

# ── Seed data ─────────────────────────────────────────────────────────────────
LEVELS = ["info", "warn", "error"]
rng = np.random.default_rng(seed=42)          # fixed seed → deterministic vectors

ids    = list(range(1, 51))                   # 1 … 50
levels = [LEVELS[(i - 1) % 3] for i in ids]  # cycles info/warn/error
seqs   = [i * 2 for i in ids]                # seq = id * 2
vectors = [
    rng.random(VECTOR_DIM).astype(np.float32).tolist()
    for _ in ids
]

seed_table = pa.table(
    {
        "id":     pa.array(ids,     type=pa.int64()),
        "level":  pa.array(levels,  type=pa.string()),
        "seq":    pa.array(seqs,    type=pa.int32()),
        "vector": pa.array(vectors, type=pa.list_(pa.float32(), VECTOR_DIM)),
    },
    schema=SCHEMA,
)

# ── Connect & (re)create table ────────────────────────────────────────────────
db = lancedb.connect(LANCEDB_URI)

# Drop existing table if present so the script is idempotent
if "logs" in db.table_names():
    db.drop_table("logs")

tbl = db.create_table("logs", data=seed_table, schema=SCHEMA)
print(f"Seeded {tbl.count_rows()} rows into '{LANCEDB_URI}/logs'.")

# ── Deletes (in order) ────────────────────────────────────────────────────────
# 1. Remove all 'warn' rows
tbl.delete("level = 'warn'")
print(f"After delete 1 (level='warn'):              {tbl.count_rows()} rows remain.")

# 2. Remove 'info' rows where seq > 60
tbl.delete("level = 'info' AND seq > 60")
print(f"After delete 2 (info AND seq>60):           {tbl.count_rows()} rows remain.")

# 3. Remove rows whose id is in {5, 9, 13}
tbl.delete("id IN (5, 9, 13)")
print(f"After delete 3 (id IN (5,9,13)):            {tbl.count_rows()} rows remain.")

# ── Export summary ────────────────────────────────────────────────────────────
total_rows = tbl.count_rows()

surviving_ids = (
    tbl.to_arrow()
       .column("id")
       .to_pylist()
)
surviving_ids_sorted = sorted(int(x) for x in surviving_ids)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result = {
    "total_rows": total_rows,
    "remaining_ids_sorted": surviving_ids_sorted,
}
OUTPUT_PATH.write_text(json.dumps(result, indent=2))

print(f"\nWrote {OUTPUT_PATH}")
print(f"  total_rows            : {total_rows}")
print(f"  remaining_ids_sorted  : {surviving_ids_sorted}")
