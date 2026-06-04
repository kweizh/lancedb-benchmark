import os
import json
import numpy as np
import pyarrow as pa
import lancedb

# ── Configuration ────────────────────────────────────────────────────────────
URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "upsert_state.json")
TABLE_NAME = "users"

# ── Schema ───────────────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("id",     pa.int64()),
    pa.field("email",  pa.string()),
    pa.field("score",  pa.float32()),
    pa.field("vector", pa.list_(pa.float32(), 8)),
])


def make_table(ids, email_fn, score_fn, vector_fn) -> pa.Table:
    """Build a PyArrow table that matches SCHEMA for the given id list."""
    ids_list   = list(ids)
    emails     = [email_fn(i)  for i in ids_list]
    scores     = [score_fn(i)  for i in ids_list]
    vectors    = [vector_fn(i) for i in ids_list]   # each is a length-8 ndarray

    flat_floats = [v for row in vectors for v in row.tolist()]

    return pa.table(
        {
            "id":     pa.array(ids_list,  type=pa.int64()),
            "email":  pa.array(emails,    type=pa.string()),
            "score":  pa.array(scores,    type=pa.float32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                          pa.array(flat_floats, type=pa.float32()), 8),
        },
        schema=SCHEMA,
    )


# ── 1. Connect and (re-)create the table ────────────────────────────────────
db = lancedb.connect(URI)
db.drop_table(TABLE_NAME, ignore_missing=True)

# Seed RNG values
seed_scores = np.random.default_rng(0).random(10).astype("float32")   # indices 0..9

seed_data = make_table(
    ids       = range(1, 11),
    email_fn  = lambda i: f"user_{i}@example.com",
    score_fn  = lambda i: float(seed_scores[i - 1]),
    vector_fn = lambda i: np.random.default_rng(100 + i).random(8).astype("float32"),
)

table = db.create_table(TABLE_NAME, data=seed_data, schema=SCHEMA, mode="overwrite")
print(f"[seed]   row count = {table.count_rows()}")   # expected: 10


# ── 2. Upsert batch 1 — UPDATE ids {2, 5, 7} ────────────────────────────────
update_data = make_table(
    ids       = [2, 5, 7],
    email_fn  = lambda i: f"updated_{i}@example.com",
    score_fn  = lambda i: np.float32(0.5 + 0.1 * i),
    vector_fn = lambda i: np.random.default_rng(200 + i).random(8).astype("float32"),
)

(table.merge_insert("id")
      .when_matched_update_all()
      .when_not_matched_insert_all()
      .execute(update_data))

print(f"[upsert1] row count = {table.count_rows()}")  # expected: 10


# ── 3. Upsert batch 2 — INSERT ids {11, 12} ─────────────────────────────────
insert_data = make_table(
    ids       = [11, 12],
    email_fn  = lambda i: f"new_{i}@example.com",
    score_fn  = lambda i: np.float32(0.5 + 0.1 * i),
    vector_fn = lambda i: np.random.default_rng(200 + i).random(8).astype("float32"),
)

(table.merge_insert("id")
      .when_matched_update_all()
      .when_not_matched_insert_all()
      .execute(insert_data))

final_count = table.count_rows()
print(f"[upsert2] row count = {final_count}")         # expected: 12
assert final_count == 12, f"Expected 12 rows, got {final_count}"


# ── 4. Export selected rows to JSON ─────────────────────────────────────────
EXPORT_IDS = {1, 2, 5, 7, 10, 11, 12}

arrow_tbl = table.to_arrow().select(["id", "email", "score"])
df = arrow_tbl.to_pydict()

rows = [
    {"id": int(i), "email": str(e), "score": float(s)}
    for i, e, s in zip(df["id"], df["email"], df["score"])
    if int(i) in EXPORT_IDS
]
rows.sort(key=lambda r: r["id"])

assert len(rows) == 7, f"Expected 7 output rows, got {len(rows)}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as fh:
    json.dump(rows, fh, indent=2)

print(f"[output]  written {len(rows)} rows → {OUTPUT_FILE}")
print(json.dumps(rows, indent=2))
