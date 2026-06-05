"""
ColBERT-style multi-vector late-interaction retrieval on LanceDB.

Each document is stored as `num_doc_tokens` rows in the `colbert_tokens` table.
At query time we issue one vector search per query token (M searches total),
convert distances to cosine similarities, compute the MaxSim contribution of
each query token over every candidate document, and return the top-k doc_ids
ranked by their summed MaxSim (late-interaction) score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lancedb

# ---------------------------------------------------------------------------
# Module-level database / table handles (opened once, reused across calls)
# ---------------------------------------------------------------------------
_DB_PATH = "/home/user/myproject/lancedb_data"
_TABLE_NAME = "colbert_tokens"

_db: lancedb.DBConnection | None = None
_tbl = None


def _get_table():
    """Return a cached LanceDB table handle, opening it on first call."""
    global _db, _tbl
    if _tbl is None:
        _db = lancedb.connect(_DB_PATH)
        _tbl = _db.open_table(_TABLE_NAME)
    return _tbl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def colbert_search(query_token_vecs: np.ndarray, k: int = 5) -> list[int]:
    """Return the top-k doc_ids ranked by ColBERT late-interaction score.

    Parameters
    ----------
    query_token_vecs:
        Float array of shape ``(M, 32)`` — one 32-d vector per query token.
    k:
        Number of results to return.

    Returns
    -------
    list[int]
        Up to *k* distinct ``doc_id`` integers in descending late-interaction
        score order.  Ties are broken by ascending ``doc_id``.
    """
    query_token_vecs = np.asarray(query_token_vecs, dtype=np.float32)
    if query_token_vecs.ndim != 2 or query_token_vecs.shape[1] != 32:
        raise ValueError(
            f"query_token_vecs must be shape (M, 32), got {query_token_vecs.shape}"
        )

    M = query_token_vecs.shape[0]
    tbl = _get_table()

    # Accumulate per-(doc_id, query_token_index) maximum similarities.
    # We collect one DataFrame per query token, then aggregate.
    per_query_frames: list[pd.DataFrame] = []

    for i in range(M):
        q_vec = query_token_vecs[i]  # shape (32,), dtype float32

        # LanceDB vector search — returns _distance = 1 - cosine_similarity
        results: pd.DataFrame = (
            tbl.search(q_vec)
            .distance_type("cosine")
            .limit(50)
            .to_pandas()
        )

        # Convert distance → similarity and tag with the query token index
        results = results[["doc_id", "token_idx", "_distance"]].copy()
        results["similarity"] = 1.0 - results["_distance"]
        results["query_token"] = i

        per_query_frames.append(results)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    # For each query token i, compute MaxSim(q_i, doc):
    #   = max cosine similarity over all of doc's token rows that appeared
    #     in the top-50 result set for q_i.
    # Documents missing from a particular query's result set contribute 0.
    #
    # Late-interaction score = Σ_i  MaxSim(q_i, doc)
    # ------------------------------------------------------------------
    doc_scores: dict[int, float] = {}

    for frame in per_query_frames:
        # Best similarity per document for this query token
        max_sim_per_doc: pd.Series = (
            frame.groupby("doc_id")["similarity"].max()
        )

        for doc_id, sim in max_sim_per_doc.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + float(sim)

    if not doc_scores:
        return []

    # Sort: descending score, ties broken by ascending doc_id
    ranked = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))

    return [int(doc_id) for doc_id, _ in ranked[:k]]
