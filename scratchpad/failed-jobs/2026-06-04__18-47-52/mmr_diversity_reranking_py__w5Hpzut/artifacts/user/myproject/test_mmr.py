import numpy as np
import lancedb
from solution import mmr_search, get_db_path, get_table_name

db = lancedb.connect(get_db_path())
table = db.open_table(get_table_name())
data = table.to_pandas()
# Create a query that is the average of centroids from clusters 0 to 8
centroids = []
for c in range(10):
    cluster_data = data[data['cluster_id'] == c]
    centroids.append(np.mean(np.stack(cluster_data['vector']), axis=0))

query = sum(centroids[:8])
print("lambda 1.0:", mmr_search(query, k=10, lambda_=1.0))
print("lambda 0.3:", mmr_search(query, k=10, lambda_=0.3))
print("lambda 0.7:", mmr_search(query, k=10, lambda_=0.7))

res_03 = mmr_search(query, k=10, lambda_=0.3)
clusters_03 = set([x // 12 for x in res_03])
print("clusters 0.3:", len(clusters_03))

res_07 = mmr_search(query, k=10, lambda_=0.7)
clusters_07 = set([x // 12 for x in res_07])
print("clusters 0.7:", len(clusters_07))
