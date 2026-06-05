"""
Query expansion using WordNet synonyms + LanceDB native FTS.

expanded_search(query, k) enriches every token in *query* with up to 3
single-token WordNet synonyms, then fires a SHOULD boolean FTS query
against the pre-seeded LanceDB table, returning the top-k document IDs
sorted by FTS score descending.
"""

import os
from typing import List

import lancedb
from lancedb.query import BooleanQuery, MatchQuery, Occur
from nltk.corpus import wordnet as wn

# ---------------------------------------------------------------------------
# Module-level state: open the DB / table once and reuse across calls.
# ---------------------------------------------------------------------------
_table = None
_fts_index_created = False

MAX_SYNONYMS_PER_TOKEN = 3


def _get_table():
    """Return (and lazily initialise) the shared LanceDB table handle."""
    global _table, _fts_index_created

    if _table is None:
        uri = os.environ["LANCEDB_URI"]
        table_name = os.environ["LANCEDB_TABLE"]
        db = lancedb.connect(uri)
        _table = db.open_table(table_name)

    if not _fts_index_created:
        # replace=False means it is a no-op if the index already exists.
        try:
            _table.create_fts_index("content", use_tantivy=False, replace=False)
        except Exception:
            # Index may already exist; ignore the error.
            pass
        _fts_index_created = True

    return _table


def _expand_token(token: str) -> List[str]:
    """
    Return a deduplicated list [token] + up to MAX_SYNONYMS_PER_TOKEN
    single-token WordNet synonyms (lowercased, no underscores/spaces).
    """
    token = token.lower()
    seen: set = {token}
    synonyms: List[str] = [token]

    for synset in wn.synsets(token):
        if len(synonyms) - 1 >= MAX_SYNONYMS_PER_TOKEN:
            break
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            # Keep only single-token lemmas (no _ or whitespace)
            if "_" in name or " " in name:
                continue
            if name not in seen:
                seen.add(name)
                synonyms.append(name)
                if len(synonyms) - 1 >= MAX_SYNONYMS_PER_TOKEN:
                    break

    return synonyms


def expanded_search(query: str, k: int = 10) -> List[int]:
    """
    Perform FTS search with WordNet query expansion.

    Parameters
    ----------
    query : str
        The raw query string (whitespace-separated tokens).
    k : int
        Maximum number of results to return.

    Returns
    -------
    list[int]
        Top-k document IDs ordered by FTS score descending.
    """
    table = _get_table()

    # Build the expanded term list for every token in the query.
    all_terms: List[str] = []
    for token in query.lower().split():
        all_terms.extend(_expand_token(token))

    # Deduplicate while preserving order (original token first per group).
    seen: set = set()
    unique_terms: List[str] = []
    for term in all_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    # Construct a SHOULD (OR) boolean FTS query over all expanded terms.
    if len(unique_terms) == 1:
        # Single term — plain match query is sufficient.
        fts_query = MatchQuery(unique_terms[0], "content")
    else:
        clauses = [
            (Occur.SHOULD, MatchQuery(term, "content")) for term in unique_terms
        ]
        fts_query = BooleanQuery(clauses)

    results = table.search(fts_query, query_type="fts").limit(k).to_list()

    return [int(row["id"]) for row in results]
