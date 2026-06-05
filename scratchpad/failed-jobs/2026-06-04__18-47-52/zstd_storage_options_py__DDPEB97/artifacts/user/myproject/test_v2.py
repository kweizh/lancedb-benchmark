import lancedb
import pyarrow as pa
import numpy as np
import json

db = lancedb.connect("/home/user/myproject/lancedb_data", storage_options={"new_table_data_storage_version": "2.0"})
data = [{"id": i, "payload": "Repeating text " * 100, "embedding": [0.0]*32} for i in range(100)]
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string(), metadata={"lance-encoding:compression": "zstd"}),
    pa.field("embedding", pa.list_(pa.float32(), 32))
])
db.create_table("test_v2", data=data, schema=schema, mode="overwrite")
