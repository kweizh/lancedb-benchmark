import lancedb
import os
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
res = table.search("fast", query_type="fts").limit(10).to_list()
print([r['id'] for r in res])
