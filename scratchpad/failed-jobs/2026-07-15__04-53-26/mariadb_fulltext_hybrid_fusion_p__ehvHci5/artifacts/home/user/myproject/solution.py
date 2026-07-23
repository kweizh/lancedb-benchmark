"""Hybrid keyword + vector search fused with Reciprocal Rank Fusion.

Reads connection settings from environment variables, queries MariaDB for
keyword relevance (FULLTEXT) and LanceDB for semantic relevance (cosine
distance), then combines the two ranked lists into a single ranking using
Reciprocal Rank Fusion with the standard constant ``60``.
"""

from __future__ import annotations

import os

import lancedb
import pymysql


# Standard Reciprocal Rank Fusion constant.
RRF_CONSTANT = 60

# A generous upper bound used when asking LanceDB for its nearest neighbors.
# The limit just has to be >= the corpus size to return every row; any ties
# in distance are broken deterministically below.
_LANCE_LIMIT = 10_000_000


def hybrid_search(query_text: str, query_vector: list[float], k: int) -> list[dict]:
    """Return the top-``k`` documents that best match ``query_text`` and ``query_vector``.

    Args:
        query_text: text passed to MariaDB ``MATCH ... AGAINST`` keyword search.
        query_vector: embedding passed to LanceDB cosine-distance vector search.
        k: maximum number of fused results to return.

    Returns:
        A list of at most ``k`` ``{"id": int, "score": float}`` dictionaries
        sorted by the fused RRF score descending, with ties broken by ``id``
        ascending. The function is deterministic for identical inputs.
    """
    if k <= 0:
        return []

    table_name = f"docs_{os.environ['ZEALT_RUN_ID']}"

    # 1. Keyword ranking from MariaDB FULLTEXT index.
    kw_ids = _keyword_search(query_text, table_name)

    # 2. Semantic ranking from LanceDB cosine-distance search.
    vec_ids = _vector_search(query_vector, table_name)

    # 3. Reciprocal Rank Fusion with constant = 60.
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(kw_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)
    for rank, doc_id in enumerate(vec_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)

    # 4. Sort by fused score descending, ties broken by id ascending.
    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    return [
        {"id": int(doc_id), "score": float(score)}
        for doc_id, score in sorted_items[:k]
    ]


def _keyword_search(query_text: str, table_name: str) -> list[int]:
    """Run MariaDB ``MATCH ... AGAINST`` and return ids in relevance order."""
    conn = pymysql.connect(
        host=os.environ["MARIADB_HOST"],
        port=int(os.environ["MARIADB_PORT"]),
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        database=os.environ["MARIADB_DATABASE"],
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            sql = (
                f"SELECT id, "
                f"MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) AS rel "
                f"FROM `{table_name}` "
                f"WHERE MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) "
                f"ORDER BY rel DESC, id ASC"
            )
            cur.execute(sql, (query_text, query_text))
            return [int(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _vector_search(query_vector: list[float], table_name: str) -> list[int]:
    """Run LanceDB cosine-distance search and return ids in similarity order."""
    db = lancedb.connect(os.environ["LANCEDB_PATH"])
    table = db.open_table(table_name)

    df = (
        table.search(query_vector, vector_column_name="vector")
        .metric("cosine")
        .limit(_LANCE_LIMIT)
        .to_pandas()
    )

    # Cosine distance: smaller == more similar. Break distance ties by id so
    # the ranking is fully deterministic across calls.
    df = df.sort_values(by=["_distance", "id"], kind="mergesort").reset_index(
        drop=True
    )
    return [int(value) for value in df["id"].tolist()]