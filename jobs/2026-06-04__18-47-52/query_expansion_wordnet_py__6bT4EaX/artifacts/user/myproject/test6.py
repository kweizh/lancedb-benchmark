import lancedb
import os
from solution import expanded_search
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
print(expanded_search("fast car", 10))
