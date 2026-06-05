import os
import lancedb
import numpy as np

uri = os.environ.get('LANCEDB_URI')
table_name = os.environ.get('LANCEDB_TABLE')
db = lancedb.connect(uri)
table = db.open_table(table_name)
df = table.to_pandas()
v = df.iloc[0]['vector']
s = table.search(v)
print(hasattr(s, "distance_type"))
print(hasattr(s, "metric"))
