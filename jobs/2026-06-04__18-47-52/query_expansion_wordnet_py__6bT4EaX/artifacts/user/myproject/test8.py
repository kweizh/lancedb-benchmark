import lancedb
import os
from solution import expanded_search
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
print(table.search("blue", query_type="fts").limit(10).to_list())
print(expanded_search("blue car", 10))
