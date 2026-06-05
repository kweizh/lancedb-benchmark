import os
import json
import numpy as np
import lancedb
import datetime

def compute_ground_truth(data, queries, k=10):
    gt = []
    for q in queries:
        dists = np.sum((data - q)**2, axis=1)
        topk = np.argsort(dists)[:k]
        gt.append(set(topk.tolist()))
    return gt

def sweep() -> dict[int, float]:
    data = np.load('/app/fixtures/data.npy')
    queries = np.load('/app/fixtures/queries.npy')
    
    gt = compute_ground_truth(data, queries, k=10)
    
    db_path = '/home/user/myproject/lancedb_data/'
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    
    run_id = os.environ.get('ZEALT_RUN_ID', 'default')
    table_name = f"sweep_table_{run_id}"
    
    records = [{"id": i, "vector": row.tolist()} for i, row in enumerate(data)]
    
    if table_name in db.table_names():
        db.drop_table(table_name)
        
    table = db.create_table(table_name, data=records)
    
    results = {}
    
    for m in [4, 8, 16]:
        index_name = "my_index"
        table.create_index(
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=16,
            num_sub_vectors=m,
            name=index_name,
            metric="l2"
        )
        table.wait_for_index([index_name], timeout=datetime.timedelta(seconds=60))
        
        recalls = []
        for i, q in enumerate(queries):
            res = table.search(q.tolist()).limit(10).nprobes(16).to_list()
            res_ids = set([r["id"] for r in res])
            gt_ids = gt[i]
            recall = len(res_ids.intersection(gt_ids)) / 10.0
            recalls.append(recall)
            
        results[m] = float(np.mean(recalls))
        
        table.drop_index(index_name)
        
    return results
