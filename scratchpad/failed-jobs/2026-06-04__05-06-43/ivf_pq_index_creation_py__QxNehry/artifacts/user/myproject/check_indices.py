import os
import lancedb

uri = os.environ.get("LANCEDB_URI", "/workspace/db")
db = lancedb.connect(uri)
table = db.open_table("embeddings")
indices = table.list_indices()
for idx in indices:
    print(dir(idx))
    print("idx:", idx)
    print("type:", type(idx))
    print("index_type:", getattr(idx, "index_type", None))
    print("columns:", getattr(idx, "columns", None))
