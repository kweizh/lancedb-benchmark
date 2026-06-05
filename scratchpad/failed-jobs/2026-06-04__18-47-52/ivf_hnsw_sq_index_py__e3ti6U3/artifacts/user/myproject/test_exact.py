import solution
import numpy as np

rng = np.random.default_rng(42)
queries = rng.standard_normal((100, 128)).astype("float32")

all_data = solution.table.search().limit(1024).to_list()
vectors = np.array([row["vector"] for row in all_data])
ids = np.array([row["id"] for row in all_data])

def cosine_dist_matrix(Q, V):
    Q_norm = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    V_norm = V / np.linalg.norm(V, axis=1, keepdims=True)
    return 1.0 - np.dot(Q_norm, V_norm.T)

dists = cosine_dist_matrix(queries, vectors)
exact_top10_idx = np.argsort(dists, axis=1)[:, :10]
exact_top10_ids = ids[exact_top10_idx]

recall = 0
for i, q in enumerate(queries):
    res = solution.search(q.tolist(), 10, 8)
    res_ids = set(r["id"] for r in res)
    exact_ids = set(exact_top10_ids[i])
    recall += len(res_ids.intersection(exact_ids)) / 10.0

print(f"Average recall: {recall / 100.0}")
