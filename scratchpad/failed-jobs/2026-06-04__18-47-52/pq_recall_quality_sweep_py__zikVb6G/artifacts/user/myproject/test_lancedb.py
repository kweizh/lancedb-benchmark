import lancedb
import numpy as np
db = lancedb.connect('./test_db')
table = db.open_table("test_table")
import builtins
print(builtins.help(table.wait_for_index))
