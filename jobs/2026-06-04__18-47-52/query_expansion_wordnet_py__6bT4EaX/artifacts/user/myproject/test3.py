import lancedb
import os
from solution import expanded_search
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)

res1 = table.search("car", query_type="fts").limit(20).to_list()
print("Plain car:", [r['id'] for r in res1])

print("Expanded search:", expanded_search("car", 20))
