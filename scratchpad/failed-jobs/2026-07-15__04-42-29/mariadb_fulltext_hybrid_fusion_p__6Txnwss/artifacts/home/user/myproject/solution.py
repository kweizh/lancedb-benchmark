"""
Hybrid Keyword + Vector Search with Reciprocal Rank Fusion (RRF).

Queries MariaDB (FULLTEXT keyword search) and LanceDB (vector/semantic search)
over the same corpus and fuses the two ranked lists using RRF with constant 60.
"""

from __future__ import annotations

import os
from typing import Any

import lancedb
import pymysql

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RRF_K = 60  # RRF smoothing constant


def _get_table_name() -> str:
    run_id = os.environ["ZEALT_RUN_ID"]
    return f"docs_{run_id}"


def _mariadb_conn() -> pymysql.Connection:
    return pymysql.connect(
        host=os.environ["MARIADB_HOST"],
        port=int(os.environ["MARIADB_PORT"]),
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        database=os.environ["MARIADB_DATABASE"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _keyword_search(query_text: str, table_name: str) -> list[int]:
    """Return document ids ordered by FULLTEXT relevance (best first)."""
    sql = (
        f"SELECT id "
        f"FROM `{table_name}` "
        f"WHERE MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) "
        f"ORDER BY MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) DESC"
    )
    with _mariadb_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_text, query_text))
            rows = cur.fetchall()
    return [int(row["id"]) for row in rows]


def _vector_search(query_vector: list[float], table_name: str) -> list[int]:
    """Return document ids ordered by cosine similarity (most similar first)."""
    db_path = os.environ["LANCEDB_PATH"]
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    results = (
        table.search(query_vector)
        .metric("cosine")
        .limit(None)          # retrieve all rows so ranking is global
        .to_pandas()
    )
    return [int(doc_id) for doc_id in results["id"].tolist()]


def _rrf_fuse(
    keyword_ids: list[int],
    vector_ids: list[int],
    k: int,
) -> list[dict[str, Any]]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.

    score(d) = sum over lists containing d of 1 / (RRF_K + rank_in_list)

    Returns up to k dicts with keys 'id' (int) and 'score' (float),
    sorted by score descending then id ascending.
    """
    scores: dict[int, float] = {}

    for rank, doc_id in enumerate(keyword_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)

    for rank, doc_id in enumerate(vector_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)

    sorted_results = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [{"id": doc_id, "score": score} for doc_id, score in sorted_results[:k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    k: int,
) -> list[dict]:
    """
    Hybrid keyword + vector search fused with Reciprocal Rank Fusion.

    Parameters
    ----------
    query_text:
        The text query sent to MariaDB's FULLTEXT index.
    query_vector:
        The embedding vector sent to LanceDB's vector index.
    k:
        Maximum number of results to return.

    Returns
    -------
    A list of at most k dicts, each with:
      - 'id'    (int)   : document identifier shared across both stores.
      - 'score' (float) : RRF fused score; higher is better.
    Sorted by score descending; ties broken by id ascending.
    """
    table_name = _get_table_name()
    keyword_ids = _keyword_search(query_text, table_name)
    vector_ids = _vector_search(query_vector, table_name)
    return _rrf_fuse(keyword_ids, vector_ids, k)
