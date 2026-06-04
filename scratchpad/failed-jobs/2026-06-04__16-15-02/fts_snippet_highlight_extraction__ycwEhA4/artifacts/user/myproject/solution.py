"""Search helper that returns BM25-ranked results with highlighted snippets."""

import os
import re

import lancedb


def _find_match(body: str, query: str):
    """Return (match_start, match_len) for the best match in body, or None."""
    # Strategy 1: exact case-insensitive substring match of the full query
    pos = body.lower().find(query.lower())
    if pos >= 0:
        return pos, len(query)

    # Strategy 2: first whole-word match of any token in the query
    for token in query.split():
        pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
        m = pattern.search(body)
        if m:
            return m.start(), len(m.group())

    # No match found
    return None


def _build_snippet(body: str, match_pos: int, match_len: int, snippet_chars: int):
    """Build a snippet window centred on the match and wrap the match in <mark> tags."""
    body_len = len(body)

    if body_len <= snippet_chars:
        # The entire body fits in the window
        window_start = 0
        window_end = body_len
    else:
        # Centre the match within the window
        ideal_start = match_pos - (snippet_chars - match_len) // 2
        window_start = max(0, min(ideal_start, body_len - snippet_chars))
        window_end = window_start + snippet_chars

    window_text = body[window_start:window_end]

    # Positions of the match relative to the window start
    mark_start = match_pos - window_start
    mark_end = mark_start + match_len

    snippet = (
        window_text[:mark_start]
        + "<mark>"
        + window_text[mark_start:mark_end]
        + "</mark>"
        + window_text[mark_end:]
    )

    return snippet, window_start


def search_with_snippets(query: str, k: int, snippet_chars: int = 120):
    """Search the LanceDB FTS index and return results with highlighted snippets.

    Parameters
    ----------
    query : str
        The full-text search query.
    k : int
        Maximum number of results to return.
    snippet_chars : int
        Maximum number of original-body characters in the snippet window.

    Returns
    -------
    list[dict]
        Each dict has keys: id (int), score (float), snippet (str),
        snippet_offset (int).
    """
    db = lancedb.connect("/home/user/myproject/data")
    table = db.open_table(os.environ["LANCE_TABLE"])

    rows = table.search(query, query_type="fts").limit(k).to_list()

    results = []
    for row in rows:
        body = row["body"]
        match = _find_match(body, query)

        if match is not None:
            match_pos, match_len = match
            snippet, snippet_offset = _build_snippet(
                body, match_pos, match_len, snippet_chars
            )
        else:
            # Fallback: leading characters with no markup
            window_end = min(snippet_chars, len(body))
            snippet = body[:window_end]
            snippet_offset = 0

        results.append(
            {
                "id": row["id"],
                "score": row["_score"],
                "snippet": snippet,
                "snippet_offset": snippet_offset,
            }
        )

    return results