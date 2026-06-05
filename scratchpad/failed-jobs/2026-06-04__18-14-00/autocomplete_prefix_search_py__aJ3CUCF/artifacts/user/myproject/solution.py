import os
import lancedb


def autocomplete(prefix: str, k: int = 10) -> list[dict]:
    """Return autocomplete suggestions for the given prefix.

    Strategy:
    1. Case-insensitive SQL prefix match on the movies table.
    2. If fewer than k matches, fill the remainder with a vector-search
       using the precomputed embedding for the lowercased prefix, excluding
       ids already returned by the prefix match.
    """
    uri = os.environ["LANCE_DB_URI"]
    movies_table_name = os.environ["MOVIES_TABLE"]
    prefix_table_name = os.environ["PREFIX_TABLE"]

    db = lancedb.connect(uri)
    movies_tbl = db.open_table(movies_table_name)
    prefix_tbl = db.open_table(prefix_table_name)

    lower_prefix = prefix.lower()

    # --- Step 1: SQL prefix match (case-insensitive via title_lower column) ---
    sql_filter = f"title_lower LIKE '{_escape_like(lower_prefix)}%'"
    prefix_rows = (
        movies_tbl
        .search()
        .where(sql_filter, prefilter=True)
        .select(["id", "title", "popularity"])
        .limit(movies_tbl.count_rows())  # fetch all matches so we can sort
        .to_list()
    )

    # Sort by popularity descending and keep top-k
    prefix_rows.sort(key=lambda r: r["popularity"], reverse=True)
    prefix_results = prefix_rows[:k]

    for row in prefix_results:
        row["source"] = "prefix"

    m = len(prefix_results)
    if m >= k:
        return prefix_results

    # --- Step 2: Vector-search fallback for the remaining (k - m) slots ---
    need = k - m

    # Look up the precomputed embedding for the lowercased prefix
    pv_rows = (
        prefix_tbl
        .search()
        .where(f"prefix = '{_escape_sql(lower_prefix)}'", prefilter=True)
        .limit(1)
        .to_list()
    )

    if not pv_rows:
        # No embedding found – return what we have
        return prefix_results

    query_vector = list(pv_rows[0]["vector"])

    # Build exclusion filter for ids already in prefix results
    excluded_ids = [row["id"] for row in prefix_results]
    if excluded_ids:
        ids_literal = ", ".join(str(i) for i in excluded_ids)
        excl_filter = f"id NOT IN ({ids_literal})"
    else:
        excl_filter = None

    searcher = (
        movies_tbl
        .search(query_vector)
        .select(["id", "title", "popularity"])
        .limit(need)
    )

    if excl_filter:
        searcher = searcher.where(excl_filter, prefilter=True)

    semantic_rows = searcher.to_list()

    for row in semantic_rows:
        row["source"] = "semantic"
        # Remove LanceDB internal distance column if present
        row.pop("_distance", None)

    return prefix_results + semantic_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape_like(value: str) -> str:
    """Escape special LIKE characters in a pattern value."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")


def _escape_sql(value: str) -> str:
    """Escape single quotes for use in a SQL string literal."""
    return value.replace("'", "''")
