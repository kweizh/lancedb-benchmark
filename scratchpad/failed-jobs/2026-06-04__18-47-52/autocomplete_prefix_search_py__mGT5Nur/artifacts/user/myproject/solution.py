import os
import lancedb

def autocomplete(prefix: str, k: int = 10) -> list[dict]:
    uri = os.environ.get("LANCE_DB_URI")
    movies_table_name = os.environ.get("MOVIES_TABLE")
    prefix_table_name = os.environ.get("PREFIX_TABLE")

    db = lancedb.connect(uri)
    movies_table = db.open_table(movies_table_name)
    prefix_table = db.open_table(prefix_table_name)

    prefix_lower = prefix.lower()
    safe_prefix = prefix_lower.replace("'", "''")
    
    # 1. SQL Prefix match
    prefix_matches = movies_table.search().where(f"title_lower LIKE '{safe_prefix}%'").limit(1000).to_list()
    
    # Sort by popularity descending
    prefix_matches.sort(key=lambda x: x['popularity'], reverse=True)
    
    M = len(prefix_matches)
    
    results = []
    
    if M >= k:
        for m in prefix_matches[:k]:
            results.append({
                "id": m["id"],
                "title": m["title"],
                "popularity": m["popularity"],
                "source": "prefix"
            })
        return results
    else:
        for m in prefix_matches:
            results.append({
                "id": m["id"],
                "title": m["title"],
                "popularity": m["popularity"],
                "source": "prefix"
            })
            
        # Semantic search
        # Fetch all prefix vectors (only 50 rows) to avoid guessing the column name
        prefix_data = prefix_table.search().limit(1000).to_list()
        prefix_vector = None
        for row in prefix_data:
            for k_col, v_col in row.items():
                if isinstance(v_col, str) and v_col == prefix_lower:
                    # we found the matching prefix string
                    # the vector column is usually named 'vector'
                    # if it's named something else, we can find the list/array
                    for col_name, col_val in row.items():
                        if isinstance(col_val, (list, tuple)) or type(col_val).__name__ == 'ndarray':
                            prefix_vector = col_val
                            break
                    break
            if prefix_vector is not None:
                break
                
        if prefix_vector is None:
            # Fallback if we couldn't find the prefix
            return results
            
        # We need to exclude ids already returned.
        exclude_ids = [m['id'] for m in prefix_matches]
        
        search_query = movies_table.search(prefix_vector).limit(k - M)
        if exclude_ids:
            ids_str = ", ".join(map(str, exclude_ids))
            search_query = search_query.where(f"id NOT IN ({ids_str})", prefilter=True)
            
        semantic_matches = search_query.to_list()
        
        for m in semantic_matches:
            results.append({
                "id": m["id"],
                "title": m["title"],
                "popularity": m["popularity"],
                "source": "semantic"
            })
            
        return results
