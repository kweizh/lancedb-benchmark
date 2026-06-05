import lancedb
import numpy as np
db = lancedb.connect('./test_db2')
data = [{"id": i, "vector": np.random.rand(64).astype(np.float32).tolist()} for i in range(1024)]
table = db.create_table("test_table2", data=data)
import builtins
print(builtins.help(table.wait_for_index))
