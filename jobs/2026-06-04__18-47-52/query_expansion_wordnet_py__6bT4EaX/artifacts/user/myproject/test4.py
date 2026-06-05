import lancedb
import os
uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("LANCEDB_TABLE")
db = lancedb.connect(uri)
table = db.open_table(table_name)
docs = table.search().to_list()
for d in docs:
    if d['id'] in [8, 7, 6, 12]:
        print(d)
