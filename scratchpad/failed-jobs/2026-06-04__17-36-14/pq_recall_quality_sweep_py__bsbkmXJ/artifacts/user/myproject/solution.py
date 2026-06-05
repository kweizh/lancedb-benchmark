import os
import lancedb
import numpy as np
import datetime

def sweep() -> dict[int, float]:
    """
    Runs a sweep over num_sub_vectors in [4, 8, 16] using IVF_PQ index in LanceDB,
    and returns a dictionary mapping num_sub_vectors to mean recall@10.
    """
    # Load dataset and query vectors
    dataset = np.load('/app/fixtures/data.npy')
    queries = np.load('/app/fixtures/queries.npy')
    
    # Compute brute-force ground truth using NumPy (squared L2 distance)
    ground_truth = []
    for q in queries:
        diff = dataset - q
        dists = np.sum(diff ** 2, axis=1)
        top10 = np.argsort(dists)[:10]
        ground_truth.append(set(top10))
        
    # Connect to local LanceDB database
    db_path = '/home/user/myproject/lancedb_data/'
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    
    # Read the ZEALT_RUN_ID environment variable
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    
    results = {}
    for m in [4, 8, 16]:
        # Formulate table name with ZEALT_RUN_ID suffix for parallel-run safety
        table_name = f"dataset_m{m}_{run_id}" if run_id else f"dataset_m{m}"
        
        # Drop table if it already exists to guarantee a clean state
        if table_name in db.table_names():
            db.drop_table(table_name)
            
        # Format records with stable 'id' and 'vector'
        records = [{"id": int(i), "vector": dataset[i].tolist()} for i in range(len(dataset))]
        tbl = db.create_table(table_name, data=records)
        
        # Create IVF_PQ index
        tbl.create_index(
            metric="l2",
            num_partitions=16,
            num_sub_vectors=m,
            vector_column_name="vector",
            name="vector_idx"
        )
        
        # Wait for index build completion
        tbl.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=300))
        
        # Evaluate recall@10 over all queries
        recalls = []
        for i, q in enumerate(queries):
            # Query the table with nprobes=16
            results_list = tbl.search(q.tolist()).limit(10).nprobes(16).to_list()
            candidate_ids = set(row["id"] for row in results_list)
            
            # Compute intersection and recall@10
            intersection = candidate_ids.intersection(ground_truth[i])
            recall = len(intersection) / 10.0
            recalls.append(recall)
            
        mean_recall = sum(recalls) / len(recalls)
        results[m] = mean_recall
        
    return results
