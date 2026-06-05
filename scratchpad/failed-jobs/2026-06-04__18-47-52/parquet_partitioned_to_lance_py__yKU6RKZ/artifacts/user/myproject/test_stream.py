import pyarrow.dataset as ds
import lancedb
db = lancedb.connect("/home/user/myproject/lancedb")
dataset = ds.dataset("/home/user/myproject/parquet_dataset", format="parquet", partitioning="hive")
db.create_table("test_stream", data=dataset.scanner().to_reader(), mode="overwrite")
table = db.open_table("test_stream")
print(len(table))
