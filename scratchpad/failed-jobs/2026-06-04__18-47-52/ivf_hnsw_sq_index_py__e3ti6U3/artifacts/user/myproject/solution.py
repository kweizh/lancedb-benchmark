import os
import lancedb
import numpy as np
import pyarrow as pa
from datetime import timedelta

db_path = os.environ.get("LANCE_DB_PATH", "/tmp/lancedb")
run_id = os.environ.get("ZEALT_RUN_ID", "default")
table_name = f"vectors_{run_id}"

db = lancedb.connect(db_path)

schema = pa.schema([
    pa.field("id", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), 128))
])

rng = np.random.default_rng(2026)
vectors = rng.standard_normal((1024, 128)).astype("float32")
data = [{"id": i, "vector": vectors[i].tolist()} for i in range(1024)]

table = db.create_table(table_name, data=data, schema=schema, exist_ok=True)

table.create_index(vector_column_name="vector", index_type="IVF_HNSW_SQ", metric="cosine", num_partitions=8, replace=True)

indices = table.list_indices()
index_name = indices[0].name if indices else "vector_idx"

table.wait_for_index([index_name], timeout=timedelta(seconds=120))

def search(query_vec, k, nprobes):
    return table.search(query_vec).nprobes(nprobes).limit(k).to_list()
