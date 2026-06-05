import os
import lancedb
import nltk

try:
    from nltk.corpus import wordnet as wn
    # Trigger a dummy lookup to ensure WordNet is loaded/downloaded
    wn.synsets('car')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet as wn

from lancedb.query import MatchQuery, BooleanQuery, Occur

def has_fts_index(table, column_name="content"):
    """
    Checks if a native FTS index exists on the specified column.
    """
    try:
        indices = table.list_indices()
        for idx in indices:
            if str(idx.index_type) == "FTS" and column_name in idx.columns:
                return True
    except Exception:
        pass
    return False

def expanded_search(query: str, k: int = 10) -> list[int]:
    """
    Performs full-text search with query expansion using WordNet.
    
    Parameters:
    - query: The raw search query string.
    - k: The maximum number of document IDs to return.
    
    Returns:
    - A list of integer document IDs, sorted by FTS score descending.
    """
    # 1. Read LanceDB URI and Table name from environment variables
    lancedb_uri = os.environ.get("LANCEDB_URI", "/app/lancedb_data")
    lancedb_table = os.environ.get("LANCEDB_TABLE", "docs")
    
    # 2. Connect to LanceDB and open the table
    db = lancedb.connect(lancedb_uri)
    tbl = db.open_table(lancedb_table)
    
    # 3. Ensure native FTS index exists on the 'content' column
    if not has_fts_index(tbl, "content"):
        try:
            tbl.create_fts_index("content", use_tantivy=False)
        except Exception:
            # Handle potential race conditions or pre-existing index issues gracefully
            pass
            
    # 4. Lowercase and split the query into whitespace-separated tokens
    query_lower = query.strip().lower()
    if not query_lower:
        return []
        
    tokens = query_lower.split()
    expanded_terms = []
    seen_global = set()
    
    for token in tokens:
        # Original query term must remain in the expansion
        if token not in seen_global:
            seen_global.add(token)
            expanded_terms.append(token)
            
        # Clean the token of common punctuation for WordNet lookup
        token_clean = token.strip(".,!?\"'()[]{}")
        synonyms = []
        seen_local = {token, token_clean}
        
        if token_clean:
            for synset in wn.synsets(token_clean):
                for lemma in synset.lemmas():
                    name = lemma.name().lower()
                    # Only keep single-token synonyms (no '_' or whitespace)
                    if '_' not in name and ' ' not in name:
                        if name not in seen_local:
                            seen_local.add(name)
                            synonyms.append(name)
                            if len(synonyms) == 3:
                                break
                if len(synonyms) == 3:
                    break
                    
        # Add the synonyms to the global expansion list if not already seen
        for syn in synonyms:
            if syn not in seen_global:
                seen_global.add(syn)
                expanded_terms.append(syn)
                
    if not expanded_terms:
        return []
        
    # 5. Construct an OR/SHOULD boolean FTS query over the expansion
    queries = [
        (Occur.SHOULD, MatchQuery(query=term, column='content'))
        for term in expanded_terms
    ]
    bq = BooleanQuery(queries=queries)
    
    # 6. Execute search and return top-k document IDs sorted by score descending
    results = tbl.search(bq, query_type="fts").limit(k).to_list()
    return [int(r["id"]) for r in results]
