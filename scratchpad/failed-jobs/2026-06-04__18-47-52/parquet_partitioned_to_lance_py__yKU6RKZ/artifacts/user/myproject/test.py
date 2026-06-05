import pyarrow.dataset as ds
dataset = ds.dataset("/home/user/myproject/parquet_dataset", format="parquet", partitioning="hive")
print(dataset.schema)
print(dataset.to_table().num_rows)
