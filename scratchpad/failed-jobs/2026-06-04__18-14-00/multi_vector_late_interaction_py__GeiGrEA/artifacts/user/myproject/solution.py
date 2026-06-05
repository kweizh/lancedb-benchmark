import numpy as np
import pandas as pd
import lancedb

DB_PATH = "/home/user/myproject/lancedb_data/"
TABLE_NAME = "colbert_tokens"


def colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]:
    """
    ColBERT-style multi-vector late-interaction retrieval using MaxSim.

    Parameters
    ----------
    query_token_vecs : np.ndarray
        Shape (M, 32) float array of M query token vectors.
    k : int
        Number of top documents to return.

    Returns
    -------
    list[int]
        Top-k doc_id values ordered by descending late-interaction score.
        Ties broken by ascending doc_id.
    """
    if query_token_vecs.ndim != 2 or query_token_vecs.shape[1] != 32:
        raise ValueError(
            f"query_token_vecs must be shape (M, 32), got {query_token_vecs.shape}"
        )

    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)

    M = query_token_vecs.shape[0]

    # Accumulate per-doc MaxSim scores across M query tokens.
    # doc_scores[doc_id] = sum of MaxSim(q_i, doc) for i in 0..M-1
    doc_scores: dict[int, float] = {}

    for i in range(M):
        query_vec = query_token_vecs[i].astype(np.float32)

        # One vector search per query token — retrieve top-50 rows
        results: pd.DataFrame = (
            tbl.search(query_vec)
            .distance_type("cosine")
            .limit(50)
            .to_pandas()
        )

        # Convert LanceDB cosine distance → cosine similarity
        results["_similarity"] = 1.0 - results["_distance"]

        # For this query token, compute MaxSim per candidate document:
        # take the maximum similarity among all retrieved rows belonging to
        # the same doc_id (a doc may appear multiple times if several of its
        # token rows land in the top-50).
        max_sim_per_doc = (
            results.groupby("doc_id", sort=False)["_similarity"].max()
        )

        for doc_id, max_sim in max_sim_per_doc.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + float(max_sim)

    if not doc_scores:
        return []

    # Sort by descending late-interaction score, then ascending doc_id for ties
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return [int(doc_id) for doc_id, _ in sorted_docs[:k]]
