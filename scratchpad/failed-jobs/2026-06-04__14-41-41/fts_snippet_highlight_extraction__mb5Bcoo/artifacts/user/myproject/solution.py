import os
import re
import lancedb

def search_with_snippets(query: str, k: int, snippet_chars: int = 120) -> list[dict]:
    # 1. Connect to the database and open the table
    db_path = "/home/user/myproject/data"
    table_name = os.environ["LANCE_TABLE"]
    db = lancedb.connect(db_path)
    tbl = db.open_table(table_name)
    
    # 2. Execute full-text search
    results = tbl.search(query, query_type="fts").limit(k).to_list()
    
    # 3. Process each result to extract snippet
    out_results = []
    for row in results:
        body = row.get("body", "")
        row_id = row.get("id")
        score = float(row.get("_score", 0.0))
        
        match_start = -1
        match_end = -1
        
        # Try to find the exact query string (case-insensitive) first
        stripped_query = query.strip()
        if stripped_query:
            match_start = body.lower().find(stripped_query.lower())
            if match_start != -1:
                match_end = match_start + len(stripped_query)
        
        # Fallback to whole-word tokens if exact match is not found
        if match_start == -1 and stripped_query:
            tokens = [t for t in re.split(r'\W+', stripped_query) if t]
            best_match_start = -1
            best_match_end = -1
            
            for token in tokens:
                pattern = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)
                for m in pattern.finditer(body):
                    m_start = m.start()
                    m_end = m.end()
                    if best_match_start == -1 or m_start < best_match_start:
                        best_match_start = m_start
                        best_match_end = m_end
            
            if best_match_start != -1:
                match_start = best_match_start
                match_end = best_match_end
        
        # Determine the window
        W = min(snippet_chars, len(body))
        
        if match_start != -1:
            match_len = match_end - match_start
            match_center = match_start + match_len // 2
            win_start = match_center - W // 2
            win_start = max(0, min(win_start, len(body) - W))
            win_end = win_start + W
            
            inter_start = max(match_start, win_start)
            inter_end = min(match_end, win_end)
            
            if inter_start < inter_end:
                prefix = body[win_start:inter_start]
                match_str = body[inter_start:inter_end]
                suffix = body[inter_end:win_end]
                snippet = f"{prefix}<mark>{match_str}</mark>{suffix}"
            else:
                snippet = body[win_start:win_end]
            
            snippet_offset = win_start
        else:
            win_start = 0
            win_end = W
            snippet = body[win_start:win_end]
            snippet_offset = 0
            
        out_results.append({
            "id": int(row_id),
            "score": score,
            "snippet": snippet,
            "snippet_offset": snippet_offset
        })
        
    return out_results
