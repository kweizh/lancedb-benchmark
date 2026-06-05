import lancedb
import os
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
print("Total rows:", len(table.search().to_list()))
# See what's in doc 2 and 0
docs = table.search().to_list()
for d in docs:
    if d['id'] in [2, 0, 5, 3]:
        print(d)
