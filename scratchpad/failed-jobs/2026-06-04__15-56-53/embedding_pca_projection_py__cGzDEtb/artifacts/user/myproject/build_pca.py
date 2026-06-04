"""
Build script:
  1. Read all 600 rows from LanceDB `articles` table.
  2. Fit a deterministic 16-component PCA on the 128-d embeddings.
  3. Project all 600 vectors into 16-d space.
  4. Write a new LanceDB table  articles_pca_<ZEALT_RUN_ID>  with columns:
       id (int), title (str), embedding (fixed-size list[float32, 16]), original_id (int)
  5. Persist components (16,128) and mean (128,) to /app/pca_model.npz.
"""

import os
import numpy as np
import lancedb
import pyarrow as pa
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = "/home/user/myproject/lancedb"
SRC_TABLE    = "articles"
RUN_ID       = os.environ["ZEALT_RUN_ID"]
DST_TABLE    = f"articles_pca_{RUN_ID}"
N_COMPONENTS = 16
MODEL_PATH   = "/app/pca_model.npz"
RANDOM_STATE = 42

# ── Connect & read ────────────────────────────────────────────────────────────
print(f"Connecting to LanceDB at {DB_PATH!r} …")
db = lancedb.connect(DB_PATH)

print(f"Reading table {SRC_TABLE!r} …")
src = db.open_table(SRC_TABLE)
df  = src.to_pandas()
print(f"  Loaded {len(df)} rows, columns: {df.columns.tolist()}")

ids    = df["id"].tolist()
titles = df["title"].tolist()
X      = np.stack(df["embedding"].to_numpy()).astype(np.float64)   # (600, 128)
print(f"  Embedding matrix shape: {X.shape}")

# ── Fit PCA ───────────────────────────────────────────────────────────────────
print(f"Fitting PCA (n_components={N_COMPONENTS}, random_state={RANDOM_STATE}) …")
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
pca.fit(X)

components = pca.components_.astype(np.float64)   # (16, 128)
mean_vec   = pca.mean_.astype(np.float64)          # (128,)
print(f"  components shape : {components.shape}")
print(f"  mean shape       : {mean_vec.shape}")
print(f"  explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

# ── Project all vectors ───────────────────────────────────────────────────────
print("Projecting all 600 vectors …")
X_pca = pca.transform(X).astype(np.float32)        # (600, 16)
print(f"  Projected shape: {X_pca.shape}")

# ── Save PCA model ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
np.savez(MODEL_PATH, components=components, mean=mean_vec)
print(f"  Saved PCA model to {MODEL_PATH!r}")
# Verify round-trip
loaded = np.load(MODEL_PATH)
assert loaded["components"].shape == (N_COMPONENTS, 128), "components shape mismatch"
assert loaded["mean"].shape == (128,), "mean shape mismatch"
print("  Model round-trip verification passed ✓")

# ── Create new LanceDB table ──────────────────────────────────────────────────
# Drop pre-existing table with same name (idempotent re-runs)
existing = db.table_names()
if DST_TABLE in existing:
    print(f"  Dropping existing table {DST_TABLE!r} …")
    db.drop_table(DST_TABLE)

# Build PyArrow schema with a FixedSizeList<float32>[16] embedding column
schema = pa.schema([
    pa.field("id",          pa.int64()),
    pa.field("title",       pa.utf8()),
    pa.field("embedding",   pa.list_(pa.float32(), N_COMPONENTS)),
    pa.field("original_id", pa.int64()),
])

# Build records as a PyArrow Table for precise schema control
embeddings_pa = pa.array(
    [row.tolist() for row in X_pca],
    type=pa.list_(pa.float32(), N_COMPONENTS),
)

arrow_table = pa.table(
    {
        "id":          pa.array(ids,    type=pa.int64()),
        "title":       pa.array(titles, type=pa.utf8()),
        "embedding":   embeddings_pa,
        "original_id": pa.array(ids,    type=pa.int64()),
    },
    schema=schema,
)

print(f"Creating table {DST_TABLE!r} …")
tbl = db.create_table(DST_TABLE, data=arrow_table)
row_count = tbl.count_rows()
print(f"  Rows in new table: {row_count}")
assert row_count == 600, f"Expected 600 rows, got {row_count}"

# Quick schema check
print(f"  Schema: {tbl.schema}")
print("Done ✓")
