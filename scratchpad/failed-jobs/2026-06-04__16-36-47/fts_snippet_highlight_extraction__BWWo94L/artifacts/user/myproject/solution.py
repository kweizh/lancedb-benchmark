import os
import re
import lancedb

def search_with_snippets(query: str, k: int, snippet_chars: int = 120) -> list[dict]:
    db = lancedb.connect("/home/user/myproject/data")
    table_name = os.environ.get("LANCE_TABLE", "articles")
    table = db.open_table(table_name)
    
    # Perform FTS search
    results = table.search(query, query_type="fts").limit(k).to_list()
    
    out = []
    for row in results:
        body = row.get("body", "")
        body_lower = body.lower()
        query_lower = query.lower()
        
        match_start = -1
        match_len = 0
        
        # 1. Try exact case-insensitive match
        idx = body_lower.find(query_lower)
        if idx != -1:
            match_start = idx
            match_len = len(query)
        else:
            # 2. Fall back to whole-word match of any token
            tokens = re.findall(r'\w+', query_lower)
            for token in tokens:
                m = re.search(r'\b' + re.escape(token) + r'\b', body_lower)
                if m:
                    match_start = m.start()
                    match_len = m.end() - m.start()
                    break
                    
        if match_start != -1:
            budget = snippet_chars - match_len
            left_budget = max(0, budget // 2)
            right_budget = max(0, budget - left_budget)
            
            start_idx = match_start - left_budget
            end_idx = match_start + match_len + right_budget
            
            if start_idx < 0:
                end_idx += (0 - start_idx)
                start_idx = 0
                
            if end_idx > len(body):
                start_idx = max(0, start_idx - (end_idx - len(body)))
                end_idx = len(body)
                
            if end_idx - start_idx > snippet_chars:
                end_idx = start_idx + snippet_chars
                
            match_start_in_snippet = match_start - start_idx
            match_end_in_snippet = match_start_in_snippet + match_len
            
            snippet_raw = body[start_idx:end_idx]
            
            actual_match_start = max(0, match_start_in_snippet)
            actual_match_end = min(len(snippet_raw), match_end_in_snippet)
            
            prefix = snippet_raw[:actual_match_start]
            match_str = snippet_raw[actual_match_start:actual_match_end]
            suffix = snippet_raw[actual_match_end:]
            
            snippet = f"{prefix}<mark>{match_str}</mark>{suffix}"
            snippet_offset = start_idx
        else:
            # 3. No match found
            snippet = body[:snippet_chars]
            snippet_offset = 0
            
        out.append({
            "id": row["id"],
            "score": row["_score"],
            "snippet": snippet,
            "snippet_offset": snippet_offset
        })
        
    # Sort by descending score just in case
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
