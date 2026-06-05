import solution
import numpy as np
import pyarrow as pa
import time

rng = np.random.default_rng(42)
queries = rng.standard_normal((100, 128)).astype("float32")

# Brute force
def brute_force(q):
    return solution.table.search(q).nprobes(8).limit(10).to_list()

# wait, brute force is without index, or with index using bypass?
# Let's just compare nprobes=8 and nprobes=1
recall_8 = 0
recall_1 = 0

for q in queries:
    res_8 = solution.search(q.tolist(), 10, 8)
    res_1 = solution.search(q.tolist(), 10, 1)
    
    # We can assume res_8 is near perfect since num_partitions=8 and nprobes=8
    ids_8 = set(r["id"] for r in res_8)
    ids_1 = set(r["id"] for r in res_1)
    
    recall_1 += len(ids_1.intersection(ids_8)) / 10.0

print(f"Recall@10 nprobes=1 vs nprobes=8: {recall_1 / 100.0}")
