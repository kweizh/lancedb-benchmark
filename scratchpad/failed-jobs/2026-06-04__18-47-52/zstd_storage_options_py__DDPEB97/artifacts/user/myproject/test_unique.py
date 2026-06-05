import lancedb
import pyarrow as pa
import numpy as np

db = lancedb.connect("/home/user/myproject/lancedb_data")
data = [{"id": i, "payload": f"ID:{i} " + "Repeating text " * 100, "embedding": [0.0]*32} for i in range(100)]
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 32))
])
db.create_table("test_unique", data=data, schema=schema, mode="overwrite")

schema_zstd = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string(), metadata={"lance-encoding:compression": "zstd"}),
    pa.field("embedding", pa.list_(pa.float32(), 32))
])
db.create_table("test_unique_zstd", data=data, schema=schema_zstd, mode="overwrite")
