import lancedb
import pyarrow as pa
import numpy as np

db = lancedb.connect("/home/user/myproject/lancedb_data")

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 32))
])

schema_zstd = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("payload", pa.string(), metadata={"lance-encoding:compression": "zstd"}),
    pa.field("embedding", pa.list_(pa.float32(), 32))
])

ids = list(range(100))
payloads = [f"ID:{i} " + "Repeating text " * 100 for i in range(100)]
embeddings = [[0.0]*32 for _ in range(100)]

table_default = pa.Table.from_arrays(
    [pa.array(ids), pa.array(payloads), pa.array(embeddings, pa.list_(pa.float32(), 32))],
    schema=schema
)

table_zstd = pa.Table.from_arrays(
    [pa.array(ids), pa.array(payloads), pa.array(embeddings, pa.list_(pa.float32(), 32))],
    schema=schema_zstd
)

db.create_table("test_table", data=table_default, mode="overwrite")
db.create_table("test_table_zstd", data=table_zstd, mode="overwrite")
