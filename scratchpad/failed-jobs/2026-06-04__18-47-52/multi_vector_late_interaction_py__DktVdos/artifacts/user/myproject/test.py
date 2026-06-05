import numpy as np
from solution import colbert_search

# 3 queries, 32 dimensions
q = np.random.rand(3, 32).astype(np.float32)
res = colbert_search(q, k=5)
print("Result:", res)
