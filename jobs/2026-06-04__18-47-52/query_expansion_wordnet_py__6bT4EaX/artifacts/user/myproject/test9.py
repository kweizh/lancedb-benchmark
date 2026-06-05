import lancedb
import os
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
res = table.search("blue car", query_type="fts").limit(5).to_list()
for r in res:
    print(r['id'], r['_score'])
