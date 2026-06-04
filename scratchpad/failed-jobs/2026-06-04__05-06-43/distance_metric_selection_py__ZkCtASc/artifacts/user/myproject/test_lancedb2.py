import lancedb
import numpy as np
import pyarrow as pa
import os

db = lancedb.connect("/tmp/test_db")
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), 16))
])
data = [{"id": i, "vector": np.random.randn(16).astype(np.float32)} for i in range(10)]
table = db.create_table("test", data=data, schema=schema, mode="overwrite")
query = table.search(np.random.randn(16).astype(np.float32))
print(dir(query))
