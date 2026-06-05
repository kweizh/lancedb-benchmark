import numpy as np
import lancedb
import pandas as pd


def colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]:
    """ColBERT-style MaxSim late-interaction retrieval.

    Parameters
    ----------
    query_token_vecs : np.ndarray
        (M, 32) float array of query token embeddings.
    k : int
        Number of documents to return.

    Returns
    -------
    list[int]
        Top-k doc_ids sorted by descending late-interaction score
        (ties broken by ascending doc_id).
    """
    # Connect to LanceDB and open the table
    db = lancedb.connect("/home/user/myproject/lancedb_data/")
    tbl = db.open_table("colbert_tokens")

    M = query_token_vecs.shape[0]

    # Accumulate per-query-token max similarities per doc
    # We'll collect (doc_id, similarity) pairs for each query token
    # and then group by doc_id, taking max per group, then sum across tokens.

    # Build a DataFrame with columns: doc_id, token_idx, similarity, query_idx
    all_frames = []

    for qi in range(M):
        q_vec = query_token_vecs[qi].astype(np.float32)
        # Ensure 1-D
        q_vec = q_vec.ravel()

        results = (
            tbl.search(q_vec)
            .distance_type("cosine")
            .limit(50)
            .to_pandas()
        )

        # Convert distance to cosine similarity
        results["similarity"] = 1.0 - results["_distance"]
        results["query_idx"] = qi
        all_frames.append(results[["doc_id", "token_idx", "similarity", "query_idx"]])

    combined = pd.concat(all_frames, ignore_index=True)

    # For each (query_idx, doc_id), take the max similarity across the doc's
    # retrieved token rows. This is the MaxSim for that query token.
    maxsim_per_query = combined.groupby(["query_idx", "doc_id"])["similarity"].max().reset_index()

    # Sum MaxSim values across query tokens for each document
    late_interaction_scores = maxsim_per_query.groupby("doc_id")["similarity"].sum().reset_index()
    late_interaction_scores.columns = ["doc_id", "score"]

    # Sort by descending score, break ties by ascending doc_id
    late_interaction_scores = late_interaction_scores.sort_values(
        by=["score", "doc_id"], ascending=[False, True]
    ).reset_index(drop=True)

    # Return top-k doc_ids
    top_k = late_interaction_scores.head(k)["doc_id"].tolist()

    return [int(d) for d in top_k]