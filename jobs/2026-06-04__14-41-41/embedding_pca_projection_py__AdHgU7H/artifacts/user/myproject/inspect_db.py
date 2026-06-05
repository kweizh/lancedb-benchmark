import lancedb
import pyarrow as pa

db = lancedb.connect("/home/user/myproject/lancedb/")
print("Tables in database:", db.table_names())

tbl = db.open_table("articles")
print("Table schema:")
print(tbl.schema)

# Let's read some rows
df = tbl.to_pandas()
print("Number of rows:", len(df))
print("First row:")
print(df.iloc[0])
print("Embedding type:", type(df.iloc[0]['embedding']), "shape:", df.iloc[0]['embedding'].shape)
