import lancedb
import numpy as np
import datetime
db = lancedb.connect('./test_db4')
data = [{"id": i, "vector": np.random.rand(64).astype(np.float32).tolist()} for i in range(1024)]
table = db.create_table("test_table4", data=data)
table.create_index(vector_column_name="vector", index_type="IVF_PQ", num_partitions=16, num_sub_vectors=4, name="my_index")
table.wait_for_index(["my_index"], timeout=datetime.timedelta(seconds=10))
print("Index created and waited successfully")
