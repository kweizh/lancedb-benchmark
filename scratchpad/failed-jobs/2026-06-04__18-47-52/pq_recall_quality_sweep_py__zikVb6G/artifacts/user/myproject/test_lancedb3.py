import lancedb
import numpy as np
db = lancedb.connect('./test_db3')
data = [{"id": i, "vector": np.random.rand(64).astype(np.float32).tolist()} for i in range(1024)]
table = db.create_table("test_table3", data=data)
import builtins
print(builtins.help(table.create_index))
