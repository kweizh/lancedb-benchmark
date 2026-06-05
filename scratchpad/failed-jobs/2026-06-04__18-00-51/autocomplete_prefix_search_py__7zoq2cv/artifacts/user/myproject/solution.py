import os
import lancedb
import numpy as np


def autocomplete(prefix: str, k: int = 10) -> list[dict]:
    """Return autocomplete suggestions for the given prefix.

    Strategy:
    1. SQL prefix match (case-insensitive) on the movies table.
    2. If M >= k results, return top-k by popularity (source="prefix").
    3. If M < k, return all M prefix results then fill the rest with
       vector-search results (source="semantic"), excluding ids already
       returned.
    """
    uri = os.environ["LANCE_DB_URI"]
    movies_table_name = os.environ["MOVIES_TABLE"]
    prefix_table_name = os.environ["PREFIX_TABLE"]

    db = lancedb.connect(uri)
    movies_tbl = db.open_table(movies_table_name)
    prefix_tbl = db.open_table(prefix_table_name)

    prefix_lower = prefix.lower()

    # ------------------------------------------------------------------ #
    # Step 1: SQL prefix match                                            #
    # ------------------------------------------------------------------ #
    # Use the pre-computed title_lower column for efficiency.
    like_pattern = prefix_lower.replace("'", "''")  # basic SQL escaping
    sql_filter = f"title_lower LIKE '{like_pattern}%'"

    prefix_rows = (
        movies_tbl
        .search()
        .where(sql_filter, prefilter=True)
        .limit(movies_tbl.count_rows())  # retrieve all matches
        .to_list()
    )

    # Sort by popularity descending and take top-k
    prefix_rows.sort(key=lambda r: r["popularity"], reverse=True)

    M = len(prefix_rows)

    if M >= k:
        # Return top-k prefix matches
        results = []
        for row in prefix_rows[:k]:
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "popularity": float(row["popularity"]),
                "source": "prefix",
            })
        return results

    # ------------------------------------------------------------------ #
    # Step 2: Need semantic fill-in                                       #
    # ------------------------------------------------------------------ #
    # Build the prefix portion first
    results = []
    prefix_ids = set()
    for row in prefix_rows:
        results.append({
            "id": int(row["id"]),
            "title": row["title"],
            "popularity": float(row["popularity"]),
            "source": "prefix",
        })
        prefix_ids.add(int(row["id"]))

    needed = k - M

    # Look up the precomputed embedding for this prefix
    pv_rows = (
        prefix_tbl
        .search()
        .where(f"prefix = '{prefix_lower}'", prefilter=True)
        .limit(1)
        .to_list()
    )
    if not pv_rows:
        # Prefix not found in prefix_vectors – return what we have
        return results

    query_vector = np.array(pv_rows[0]["vector"], dtype=np.float32)

    # Build exclusion filter for vector search
    if prefix_ids:
        ids_str = ", ".join(str(i) for i in prefix_ids)
        exclude_filter = f"id NOT IN ({ids_str})"
        semantic_rows = (
            movies_tbl
            .search(query_vector)
            .where(exclude_filter, prefilter=True)
            .limit(needed)
            .to_list()
        )
    else:
        semantic_rows = (
            movies_tbl
            .search(query_vector)
            .limit(needed)
            .to_list()
        )

    for row in semantic_rows:
        results.append({
            "id": int(row["id"]),
            "title": row["title"],
            "popularity": float(row["popularity"]),
            "source": "semantic",
        })

    return results
