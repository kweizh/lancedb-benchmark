"""IVF_PQ recall-quality sweep over num_sub_vectors."""

import os
import datetime
import numpy as np
import lancedb


DB_PATH = "/home/user/myproject/lancedb_data/"
DATA_PATH = "/app/fixtures/data.npy"
QUERIES_PATH = "/app/fixtures/queries.npy"

M_VALUES = [4, 8, 16]
NUM_PARTITIONS = 16
NPROBES = 16
TOP_K = 10


def _brute_force_recall(data: np.ndarray, queries: np.ndarray, top_k: int = TOP_K) -> list[set[int]]:
    """Return the set of top-k ground-truth IDs for each query using L2 distance."""
    # Pairwise squared L2: (num_queries, num_rows)
    # ||q - d||^2 = ||q||^2 - 2*q·d + ||d||^2
    q_sq = np.sum(queries ** 2, axis=1, keepdims=True)  # (Q, 1)
    d_sq = np.sum(data ** 2, axis=1, keepdims=True).T    # (1, N)
    dists = q_sq - 2 * queries @ data.T + d_sq           # (Q, N)
    top10_indices = np.argsort(dists, axis=1)[:, :top_k]
    return [set(row.tolist()) for row in top10_indices]


def sweep() -> dict[int, float]:
    """Run the IVF_PQ recall sweep over num_sub_vectors in {4, 8, 16}.

    Returns a dict mapping num_sub_vectors -> mean recall@10 over 30 queries.
    """
    run_id = os.environ.get("ZEALT_RUN_ID", "default")

    data = np.load(DATA_PATH)      # (1024, 64) float32
    queries = np.load(QUERIES_PATH) # (30, 64) float32

    # Compute brute-force ground truth
    gt_top10 = _brute_force_recall(data, queries)

    db = lancedb.connect(DB_PATH)

    results: dict[int, float] = {}

    for m in M_VALUES:
        table_name = f"ivf_pq_m{m}_{run_id}"

        # Build records: each row has an integer id and a float32 vector
        records = [
            {"id": int(i), "vector": data[i].tolist()}
            for i in range(data.shape[0])
        ]

        # Create a fresh table per m (LanceDB allows one vector index per column)
        if table_name in db.table_names():
            db.drop_table(table_name)

        table = db.create_table(table_name, records)

        index_name = f"idx_m{m}"
        table.create_index(
            metric="l2",
            num_partitions=NUM_PARTITIONS,
            num_sub_vectors=m,
            index_type="IVF_PQ",
            name=index_name,
        )
        table.wait_for_index([index_name], timeout=datetime.timedelta(seconds=120))

        # Query all 30 vectors and compute per-query recall
        recall_sum = 0.0
        for qi in range(queries.shape[0]):
            qvec = queries[qi].tolist()
            rows = table.search(qvec).limit(TOP_K).nprobes(NPROBES).to_list()
            candidate_ids = {row["id"] for row in rows}
            recall_sum += len(candidate_ids & gt_top10[qi]) / TOP_K

        results[m] = recall_sum / queries.shape[0]

    return results