"""Query expansion with WordNet synonyms + LanceDB native FTS."""

import os
from nltk.corpus import wordnet as wn
import lancedb

# Module-level cache so we don't recreate the index on every call
_db = None
_table = None
_index_created = False


def _get_table():
    """Lazily open the LanceDB connection and ensure the FTS index exists."""
    global _db, _table, _index_created

    if _db is None:
        uri = os.environ["LANCEDB_URI"]
        table_name = os.environ["LANCEDB_TABLE"]
        _db = lancedb.connect(uri)
        _table = _db.open_table(table_name)

    if not _index_created:
        # Check if the FTS index already exists to avoid a RuntimeError
        existing = [idx.name for idx in _table.list_indices()]
        if "content_idx" not in existing:
            _table.create_fts_index("content", use_tantivy=False, replace=False)
        _index_created = True

    return _table


def _expand_token(token: str) -> list[str]:
    """Return the original token plus up to 3 single-token WordNet synonyms."""
    expanded = [token]
    seen = {token}

    for synset in wn.synsets(token):
        for lemma in synset.lemmas():
            name = lemma.name()
            # Skip multi-word synonyms (contain underscore or whitespace)
            if "_" in name or " " in name:
                continue
            if name not in seen:
                seen.add(name)
                expanded.append(name)
                if len(expanded) - 1 >= 3:  # at most 3 synonyms
                    return expanded

    return expanded


def expanded_search(query: str, k: int = 10) -> list[int]:
    """Expand *query* with WordNet synonyms, run LanceDB native FTS, and return top-k doc IDs."""
    table = _get_table()

    # Lowercase the query before expansion
    query_lower = query.lower()
    tokens = query_lower.split()

    # Expand each token and collect all unique terms
    all_terms = []
    seen = set()
    for token in tokens:
        for term in _expand_token(token):
            if term not in seen:
                seen.add(term)
                all_terms.append(term)

    # Build a space-separated OR query (native FTS treats space as OR)
    expanded_query = " ".join(all_terms)

    results = table.search(expanded_query, query_type="fts").limit(k).to_list()

    return [int(row["id"]) for row in results]