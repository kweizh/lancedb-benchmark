"""
Query expansion with WordNet synonyms and LanceDB native FTS.

Exposes:
    expanded_search(query: str, k: int = 10) -> list[int]
"""

import os

import lancedb
from lancedb.query import BooleanQuery, MatchQuery, Occur
from nltk.corpus import wordnet as wn

# ---------------------------------------------------------------------------
# Module-level state: connect once, build the index at most once.
# ---------------------------------------------------------------------------
_table = None
_index_created = False


def _get_table():
    """Return the LanceDB table, connecting lazily on first call."""
    global _table
    if _table is None:
        uri = os.environ["LANCEDB_URI"]
        table_name = os.environ["LANCEDB_TABLE"]
        db = lancedb.connect(uri)
        _table = db.open_table(table_name)
    return _table


def _ensure_fts_index(table) -> None:
    """Create a native FTS index on `content` if it does not already exist."""
    global _index_created
    if _index_created:
        return
    try:
        table.create_fts_index("content", use_tantivy=False, replace=False)
    except Exception:
        # Index already exists – ignore the error.
        pass
    _index_created = True


def _expand_query(query: str, max_synonyms_per_token: int = 3) -> list[str]:
    """
    Lowercase *query*, then for each whitespace-separated token collect up to
    *max_synonyms_per_token* WordNet synonyms that are single-token (no ``_``
    or whitespace).  The original token is always included.

    Returns a flat, deduplicated list of terms.
    """
    terms: list[str] = []
    seen: set[str] = set()

    for raw_token in query.lower().split():
        # Always keep the original term.
        if raw_token not in seen:
            terms.append(raw_token)
            seen.add(raw_token)

        # Collect single-token synonyms across all senses.
        synonyms: list[str] = []
        for synset in wn.synsets(raw_token):
            for lemma in synset.lemmas():
                name = lemma.name().lower()
                # Skip multi-token lemmas and the original term itself.
                if "_" in name or " " in name:
                    continue
                if name == raw_token or name in seen:
                    continue
                synonyms.append(name)
                if len(synonyms) >= max_synonyms_per_token:
                    break
            if len(synonyms) >= max_synonyms_per_token:
                break

        for syn in synonyms:
            terms.append(syn)
            seen.add(syn)

    return terms


def expanded_search(query: str, k: int = 10) -> list[int]:
    """
    Full-text search over the seeded LanceDB table with WordNet query expansion.

    Parameters
    ----------
    query : str
        The user query (plain text).
    k : int
        Maximum number of results to return.

    Returns
    -------
    list[int]
        Document IDs ordered by FTS score descending, length ≤ k.
    """
    table = _get_table()
    _ensure_fts_index(table)

    terms = _expand_query(query)

    # Build a SHOULD (OR) boolean query over all expanded terms.
    bool_query = BooleanQuery(
        [(Occur.SHOULD, MatchQuery(term, "content")) for term in terms]
    )

    results = (
        table.search(bool_query, query_type="fts")
        .limit(k)
        .to_list()
    )

    return [int(row["id"]) for row in results]
