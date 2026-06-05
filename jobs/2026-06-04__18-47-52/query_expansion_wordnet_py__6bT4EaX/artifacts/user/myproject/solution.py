import os
import lancedb
from nltk.corpus import wordnet as wn

_index_created = False

def expanded_search(query: str, k: int = 10) -> list[int]:
    global _index_created
    
    uri = os.environ.get("LANCEDB_URI")
    table_name = os.environ.get("LANCEDB_TABLE")
    
    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    
    if not _index_created:
        try:
            table.create_fts_index("content", use_tantivy=False, replace=False)
        except Exception:
            pass # Index might already exist
        _index_created = True

    tokens = query.lower().split()
    expanded_tokens = []
    
    for token in tokens:
        expanded_tokens.append(token)
        synonyms = []
        for synset in wn.synsets(token):
            for lemma in synset.lemmas():
                name = lemma.name().lower()
                if "_" not in name and " " not in name and name != token and name not in synonyms:
                    synonyms.append(name)
        
        # Keep at most 3 synonyms per token
        expanded_tokens.extend(synonyms[:3])
    
    # space-separated terms
    final_query = " ".join(expanded_tokens)
    
    results = table.search(final_query, query_type="fts").limit(k).to_list()
    
    return [res["id"] for res in results]
