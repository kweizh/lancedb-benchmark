import lancedb
import os
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
try:
    print(table.search("", query_type="fts").limit(5).to_list())
except Exception as e:
    print("ERROR", e)
