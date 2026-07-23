"""Hybrid keyword + vector search with Reciprocal Rank Fusion (RRF).

This module bridges two independent retrieval engines over the same corpus:

* MariaDB  -- holds the document text with a FULLTEXT index (keyword relevance).
* LanceDB  -- holds precomputed embedding vectors (semantic similarity).

Both stores share an integer ``id`` key.  For a given query we ask each engine
for its ranked result list, then fuse the two lists with RRF (constant = 60).

The public entry point is :func:`hybrid_search`.
"""

from __future__ import annotations

import os
from collections import defaultdict

import lancedb
import pymysql

# RRF constant, as required by the task specification.
RRF_K = 60


def _table_name() -> str:
    """Build the shared table name from the ``ZEALT_RUN_ID`` env var."""
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    return f"docs_{run_id}"


def _keyword_ranking(query_text: str) -> list[int]:
    """Return the ordered list of document ids from the MariaDB FULLTEXT index.

    The order returned by MariaDB (sorted by MATCH relevance descending, with
    ties broken by ``id`` ascending) *is* the keyword ranking -- the first id is
    rank 1, the second is rank 2, and so on.
    """
    conn = pymysql.connect(
        host=os.environ["MARIADB_HOST"],
        port=int(os.environ["MARIADB_PORT"]),
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        database=os.environ["MARIADB_DATABASE"],
    )
    try:
        with conn.cursor() as cur:
            table = _table_name()
            # NATURAL LANGUAGE MODE is the default; positive-relevance rows are
            # returned and ordered by relevance.  We ORDER BY relevance DESC
            # then id ASC so the ranking is fully deterministic.
            sql = (
                f"SELECT id FROM `{table}` "
                f"WHERE MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) "
                f"ORDER BY MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) DESC, id ASC"
            )
            cur.execute(sql, (query_text, query_text))
            rows = cur.fetchall()
    finally:
        conn.close()

    # rows is a list of single-element tuples (id,).
    return [int(r[0]) for r in rows]


def _vector_ranking(query_vector: list[float]) -> list[int]:
    """Return the ordered list of document ids from the LanceDB vector search.

    Cosine distance is used: smaller distance == more similar, so the natural
    ascending order of the returned results is the semantic ranking (best first).
    """
    db = lancedb.connect(os.environ["LANCEDB_PATH"])
    table = db.open_table(_table_name())

    # Fetch every row so the RRF has the complete picture; this keeps the
    # fusion correct even when a document is a weak keyword match but a strong
    # vector match (or vice-versa).  count_rows() gives the exact total.
    try:
        n = int(table.count_rows())
    except Exception:
        n = 100000
    if n <= 0:
        n = 100000

    results = (
        table.search(query_vector)
        .distance_type("cosine")
        .limit(n)
        .to_list()
    )

    # Each result is a dict including the stored columns plus ``_distance``.
    # Re-sort by (distance ascending, id ascending) to guarantee a
    # deterministic order independent of LanceDB's internal tie handling.
    ranked = sorted(results, key=lambda r: (r.get("_distance", 0.0), r["id"]))
    return [int(r["id"]) for r in ranked]


def _rrf_fuse(rankings: list[list[int]], k: int) -> list[dict]:
    """Fuse several ranked id lists with Reciprocal Rank Fusion.

    ``rankings[i]`` is an ordered list of ids (best first).  A document's
    fused score is the sum, over every list it appears in, of
    ``1 / (RRF_K + rank)`` where ``rank`` is its 1-based position in that list.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (RRF_K + position)

    # Sort by score descending, ties broken by id ascending.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    return [{"id": doc_id, "score": float(score)} for doc_id, score in ordered[:k]]


def hybrid_search(query_text: str, query_vector: list[float], k: int) -> list[dict]:
    """Run a hybrid keyword + vector search and fuse with RRF.

    Parameters
    ----------
    query_text:
        The raw user query string, sent to the MariaDB FULLTEXT index.
    query_vector:
        The query embedding, sent to the LanceDB table for cosine search.
    k:
        Maximum number of fused results to return.

    Returns
    -------
    list[dict]
        At most ``k`` objects, each with keys ``id`` (int) and ``score``
        (float, the fused RRF score), sorted by ``score`` descending with
        ties broken by ``id`` ascending.  Deterministic for identical inputs.
    """
    keyword_ranking = _keyword_ranking(query_text)
    vector_ranking = _vector_ranking(query_vector)

    return _rrf_fuse([keyword_ranking, vector_ranking], k)