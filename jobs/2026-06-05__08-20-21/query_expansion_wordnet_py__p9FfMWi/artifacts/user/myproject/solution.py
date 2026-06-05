import os
import lancedb
from nltk.corpus import wordnet as wn

def get_synonyms(token: str, max_synonyms: int = 3) -> list[str]:
    """
    Look up WordNet synonyms for a token.
    Keep at most 3 synonyms per token.
    Only keep synonyms that are a single token (no '_' or whitespace).
    """
    token = token.lower()
    synonyms = set()
    for synset in wn.synsets(token):
        for lemma in synset.lemmas():
            syn_name = lemma.name().lower()
            # Requirement: no '_' or whitespace in the WordNet lemma name.
            # Also exclude the original token itself from the synonym list.
            if syn_name != token and "_" not in syn_name and " " not in syn_name:
                synonyms.add(syn_name)
                if len(synonyms) >= max_synonyms:
                    return list(synonyms)
    return list(synonyms)

def expanded_search(query: str, k: int = 10) -> list[int]:
    """
    Performs an expanded FTS search on the LanceDB table.
    """
    uri = os.environ.get("LANCEDB_URI")
    table_name = os.environ.get("LANCEDB_TABLE")
    
    if not uri or not table_name:
        raise ValueError("LANCEDB_URI and LANCEDB_TABLE environment variables must be set.")
    
    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    
    # Build a native LanceDB FTS index on the 'content' column if it doesn't exist.
    # use_tantivy=False is required for native FTS.
    try:
        table.create_fts_index("content", use_tantivy=False, replace=False)
    except Exception:
        # If it already exists or other error, we continue. 
        # LanceDB usually handles replace=False gracefully.
        pass

    # Lowercase the query before expansion.
    query_tokens = query.lower().split()
    expanded_tokens = []
    
    for token in query_tokens:
        expanded_tokens.append(token)
        syns = get_synonyms(token, max_synonyms=3)
        expanded_tokens.extend(syns)
    
    # Construct a space-separated terms query (OR/SHOULD behavior by default in FTS).
    expanded_query = " ".join(expanded_tokens)
    
    # Perform the FTS search.
    results = table.search(expanded_query, query_type="fts").limit(k).to_list()
    
    # Return the top-k document IDs as a list of Python int.
    return [int(res["id"]) for res in results]
