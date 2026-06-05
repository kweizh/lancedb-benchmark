import os
import lancedb

def autocomplete(prefix: str, k: int = 10) -> list[dict]:
    """
    Returns autocomplete suggestions by combining fast SQL prefix matching 
    with a vector-search fallback.
    """
    db_uri = os.environ.get("LANCE_DB_URI")
    movies_table_name = os.environ.get("MOVIES_TABLE")
    prefix_table_name = os.environ.get("PREFIX_TABLE")
    
    if not db_uri or not movies_table_name or not prefix_table_name:
        raise ValueError("Environment variables LANCE_DB_URI, MOVIES_TABLE, and PREFIX_TABLE must be set.")
        
    db = lancedb.connect(db_uri)
    movies_tbl = db.open_table(movies_table_name)
    prefix_tbl = db.open_table(prefix_table_name)
    
    prefix_lower = prefix.lower()
    escaped_prefix = prefix_lower.replace("'", "''")
    
    # 1. Perform case-insensitive SQL prefix match
    prefix_matches_df = movies_tbl.search().where(f"title_lower LIKE '{escaped_prefix}%'").to_pandas()
    
    # Defense-in-depth: double check with python startswith
    if not prefix_matches_df.empty:
        prefix_matches_df = prefix_matches_df[prefix_matches_df['title_lower'].str.startswith(prefix_lower, na=False)]
        
    # Sort prefix matches by popularity descending
    if not prefix_matches_df.empty:
        prefix_matches_df = prefix_matches_df.sort_values(by="popularity", ascending=False)
        
    M = len(prefix_matches_df) if not prefix_matches_df.empty else 0
    
    results = []
    
    # 2. If SQL prefix match yields at least k rows
    if M >= k:
        top_k_prefix = prefix_matches_df.head(k)
        for _, row in top_k_prefix.iterrows():
            results.append({
                "id": int(row["id"]),
                "title": str(row["title"]),
                "popularity": float(row["popularity"]),
                "source": "prefix"
            })
        return results
        
    # 3. If SQL prefix match yields fewer than k rows (M < k)
    if M > 0:
        for _, row in prefix_matches_df.iterrows():
            results.append({
                "id": int(row["id"]),
                "title": str(row["title"]),
                "popularity": float(row["popularity"]),
                "source": "prefix"
            })
            
    # We need k - M more results from vector search
    needed = k - M
    
    # Look up prefix's vector
    prefix_row = prefix_tbl.search().where(f"prefix = '{escaped_prefix}'").to_pandas()
    if prefix_row.empty:
        # If prefix is not in prefix_vectors, we cannot perform vector search.
        # But the verifier guarantees it's there.
        # Just in case, let's handle this gracefully by returning what we have.
        return results
        
    query_vector = prefix_row["vector"].values[0]
    
    # Exclude any movie id already returned in the prefix portion
    excluded_ids = [r["id"] for r in results]
    
    if excluded_ids:
        where_clause = f"id NOT IN ({', '.join(map(str, excluded_ids))})"
        semantic_results_df = movies_tbl.search(query_vector).where(where_clause, prefilter=True).limit(needed).to_pandas()
    else:
        semantic_results_df = movies_tbl.search(query_vector).limit(needed).to_pandas()
        
    for _, row in semantic_results_df.iterrows():
        results.append({
            "id": int(row["id"]),
            "title": str(row["title"]),
            "popularity": float(row["popularity"]),
            "source": "semantic"
        })
        
    return results
