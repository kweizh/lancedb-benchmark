import lancedb
import pyarrow as pa
import numpy as np

db = lancedb.connect("/home/user/myproject/lancedb_data")
data = [{"id": i, "payload": "Repeating text " * 100, "embedding": [0.0]*32} for i in range(100)]
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string(), metadata={b"lance-encoding:compression": b"zstd"}),
    pa.field("embedding", pa.list_(pa.float32(), 32))
])
db.create_table("test_v2_bytes", data=data, schema=schema, mode="overwrite", data_storage_version="2.0")
