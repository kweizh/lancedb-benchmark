import os
import lancedb
import numpy as np


def autocomplete(prefix: str, k: int = 10) -> list[dict]:
    """Return autocomplete suggestions combining SQL prefix matching with semantic fallback."""
    db_uri = os.environ["LANCE_DB_URI"]
    movies_table_name = os.environ["MOVIES_TABLE"]
    prefix_table_name = os.environ["PREFIX_TABLE"]

    db = lancedb.connect(db_uri)
    movies = db.open_table(movies_table_name)
    prefix_vectors = db.open_table(prefix_table_name)

    prefix_lower = prefix.lower()

    # Step 1: SQL prefix match (case-insensitive)
    prefix_df = (
        movies.search()
        .where(f"title_lower LIKE '{prefix_lower}%'", prefilter=True)
        .select(["id", "title", "popularity"])
        .to_pandas()
    )

    # Sort by popularity descending
    if not prefix_df.empty:
        prefix_df = prefix_df.sort_values("popularity", ascending=False).reset_index(drop=True)

    M = len(prefix_df)

    results = []

    if M >= k:
        # Enough prefix matches — return top k by popularity
        for _, row in prefix_df.head(k).iterrows():
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "popularity": float(row["popularity"]),
                "source": "prefix",
            })
        return results

    # M < k: collect all prefix matches first
    for _, row in prefix_df.iterrows():
        results.append({
            "id": int(row["id"]),
            "title": row["title"],
            "popularity": float(row["popularity"]),
            "source": "prefix",
        })

    prefix_ids = set(prefix_df["id"].tolist())
    needed = k - M

    # Step 2: Look up the prefix vector from prefix_vectors table
    pv_df = prefix_vectors.to_pandas()
    pv_row = pv_df[pv_df["prefix"] == prefix_lower]
    if pv_row.empty:
        # No vector found; return what we have
        return results

    query_vector = pv_row.iloc[0]["vector"]
    # Convert to a plain Python list / numpy array
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    elif hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    # Step 3: Vector search against movies table, excluding prefix-matched ids
    exclude_clause = ""
    if prefix_ids:
        ids_str = ",".join(str(int(i)) for i in prefix_ids)
        exclude_clause = f"id NOT IN ({ids_str})"

    search_builder = movies.search(query_vector).limit(needed)
    if exclude_clause:
        search_builder = search_builder.where(exclude_clause, prefilter=True)

    semantic_df = search_builder.select(["id", "title", "popularity"]).to_pandas()

    for _, row in semantic_df.iterrows():
        results.append({
            "id": int(row["id"]),
            "title": row["title"],
            "popularity": float(row["popularity"]),
            "source": "semantic",
        })

    return results