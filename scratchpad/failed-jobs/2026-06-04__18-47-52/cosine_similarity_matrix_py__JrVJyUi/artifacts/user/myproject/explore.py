import os
import lancedb

uri = os.environ.get('LANCEDB_URI')
table_name = os.environ.get('LANCEDB_TABLE')

db = lancedb.connect(uri)
table = db.open_table(table_name)
print(table.schema)
df = table.to_pandas()
print(df.head())
