import lancedb
import json

db = lancedb.connect("/home/user/myproject/lancedb")
table = db.open_table("pois")
print("Table columns:", table.schema)
print("Row count:", table.count_rows())
