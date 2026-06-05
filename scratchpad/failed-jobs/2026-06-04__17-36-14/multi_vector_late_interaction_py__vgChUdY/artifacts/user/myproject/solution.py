import lancedb
import numpy as np
import pandas as pd

def colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]:
    """
    ColBERT-style Multi-Vector Late-Interaction Retrieval on LanceDB.
    
    Parameters:
    - query_token_vecs: np.ndarray of shape (M, 32) representing the query token embeddings.
    - k: int, the number of top documents to retrieve.
    
    Returns:
    - list[int]: Top-k doc_id values ordered by descending late-interaction score.
    """
    # Connect to the LanceDB database
    db = lancedb.connect("/home/user/myproject/lancedb_data/")
    tbl = db.open_table("colbert_tokens")
    
    M = query_token_vecs.shape[0]
    results_per_query = []
    all_candidate_docs = set()
    
    # 1. Run one LanceDB vector search per query token using the `cosine` distance metric,
    # retrieving the top-50 (doc_id, token_idx, _distance) rows for that query token.
    for i in range(M):
        q_vec = query_token_vecs[i].astype(np.float32)
        
        # Issue search
        df = tbl.search(q_vec).distance_type("cosine").limit(50).to_pandas()
        
        # 2. Converts each `_distance` to a cosine similarity (`sim = 1 - distance`)
        df['sim'] = 1.0 - df['_distance']
        
        # 3. For every candidate document that appears in any of the M result sets,
        # computes MaxSim(q_i, doc) = max over the doc's retrieved token rows of cosine(q_i, t).
        max_sim_series = df.groupby('doc_id')['sim'].max()
        
        results_per_query.append(max_sim_series.to_dict())
        all_candidate_docs.update(max_sim_series.index.tolist())
        
    # 4. Compute late-interaction score for each candidate document:
    # The late-interaction score for a doc is the sum of MaxSim values across the M query tokens.
    # Documents whose token did not appear in a particular query's top-50 contribute 0 for that q_i.
    doc_scores = []
    for doc_id in all_candidate_docs:
        score = 0.0
        for i in range(M):
            score += results_per_query[i].get(doc_id, 0.0)
        doc_scores.append((score, int(doc_id)))
        
    # Sort by descending late-interaction score. Ties broken by ascending doc_id.
    doc_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Return the top-k doc_id values as a Python list[int]
    top_k_docs = [doc_id for score, doc_id in doc_scores[:k]]
    return top_k_docs
