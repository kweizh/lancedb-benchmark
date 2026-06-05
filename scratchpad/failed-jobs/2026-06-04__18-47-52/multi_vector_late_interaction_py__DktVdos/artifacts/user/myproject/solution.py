import lancedb
import numpy as np
import pandas as pd
from collections import defaultdict

def colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]:
    db = lancedb.connect("/home/user/myproject/lancedb_data/")
    tbl = db.open_table("colbert_tokens")
    
    M = query_token_vecs.shape[0]
    
    doc_max_sims = defaultdict(dict)
    
    for i in range(M):
        q_vec = query_token_vecs[i].astype(np.float32)
        # run search
        res = tbl.search(q_vec).distance_type("cosine").limit(50).to_pandas()
        
        for _, row in res.iterrows():
            doc_id = int(row['doc_id'])
            dist = float(row['_distance'])
            sim = 1.0 - dist
            
            if i not in doc_max_sims[doc_id]:
                doc_max_sims[doc_id][i] = sim
            else:
                if sim > doc_max_sims[doc_id][i]:
                    doc_max_sims[doc_id][i] = sim
                    
    # Now compute late-interaction score
    doc_scores = []
    for doc_id, sims_dict in doc_max_sims.items():
        score = sum(sims_dict.get(j, 0.0) for j in range(M))
        doc_scores.append((doc_id, score))
        
    # Sort by descending score, then ascending doc_id
    doc_scores.sort(key=lambda x: (-x[1], x[0]))
    
    # Return top-k
    return [x[0] for x in doc_scores[:k]]
