"""
solution.py – LanceDB FTS snippet-highlight extraction.

Exposes a single callable:
    search_with_snippets(query, k, snippet_chars=120) -> list[dict]
"""

import os
import re

import lancedb


def _find_match_offset(body: str, query: str) -> int | None:
    """Return the character offset of the first case-insensitive occurrence
    of *query* in *body*, or None if not found.

    Falls back to any single token of the query when the whole phrase is absent.
    """
    # 1. Exact phrase match (case-insensitive)
    idx = body.lower().find(query.lower())
    if idx != -1:
        return idx

    # 2. Token-level fallback: find the first whole-word match for any token
    tokens = re.split(r"\s+", query.strip())
    for token in tokens:
        if not token:
            continue
        pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
        m = pattern.search(body)
        if m:
            return m.start()

    return None


def _build_snippet(body: str, match_offset: int | None, query: str,
                   snippet_chars: int) -> tuple[str, int]:
    """Return ``(snippet_html, snippet_offset)`` for the given *body*.

    *snippet_html* has at most *snippet_chars* characters of original body
    text, with the matched term wrapped in ``<mark>…</mark>`` tags.
    *snippet_offset* is the index in *body* where the window begins.
    """
    body_len = len(body)

    if match_offset is None:
        # No match at all – return the leading window with no markup.
        window_end = min(snippet_chars, body_len)
        return body[:window_end], 0

    # Determine the length of the matched text in the original body.
    # We use the *query* as the phrase first; if that failed we fall back to a
    # single token (re-detect which one actually matched at match_offset).
    if body.lower()[match_offset:match_offset + len(query)].lower() == query.lower():
        match_len = len(query)
    else:
        # Token fallback – figure out how long the token at match_offset is.
        tokens = re.split(r"\s+", query.strip())
        match_len = 0
        for token in tokens:
            if not token:
                continue
            pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
            m = pattern.match(body[match_offset:])
            if m:
                match_len = m.end() - m.start()
                break
        if match_len == 0:
            # Last resort: treat match_offset..next whitespace as the match
            rest = body[match_offset:]
            ws = re.search(r"\s", rest)
            match_len = ws.start() if ws else len(rest)

    # Center the window on the matched span.
    half = snippet_chars // 2
    win_start = match_offset - half
    win_start = max(0, win_start)

    win_end = win_start + snippet_chars
    if win_end > body_len:
        win_end = body_len
        win_start = max(0, win_end - snippet_chars)

    # Clamp match boundaries to the window.
    match_end = match_offset + match_len

    # Build the snippet: pre + <mark>term</mark> + post
    pre = body[win_start:match_offset]
    term = body[match_offset:match_end]
    post = body[match_end:win_end]

    snippet = f"{pre}<mark>{term}</mark>{post}"
    return snippet, win_start


def search_with_snippets(query: str, k: int, snippet_chars: int = 120) -> list[dict]:
    """Execute a full-text search and return snippet-annotated results.

    Parameters
    ----------
    query:
        The search query string.
    k:
        Maximum number of results to return.
    snippet_chars:
        The maximum number of *original body characters* (excluding markup
        tags) that the snippet window may span.

    Returns
    -------
    list[dict] sorted by descending BM25 score, each with keys:
        id, score, snippet, snippet_offset
    """
    table_name = os.environ["LANCE_TABLE"]
    db = lancedb.connect("/home/user/myproject/data")
    table = db.open_table(table_name)

    hits = table.search(query, query_type="fts").limit(k).to_list()

    results: list[dict] = []
    for hit in hits:
        body: str = hit["body"]
        match_offset = _find_match_offset(body, query)
        snippet, snippet_offset = _build_snippet(body, match_offset, query, snippet_chars)

        results.append(
            {
                "id": int(hit["id"]),
                "score": float(hit["_score"]),
                "snippet": snippet,
                "snippet_offset": snippet_offset,
            }
        )

    # LanceDB FTS already returns results sorted by descending score, but
    # we sort explicitly to guarantee the contract.
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
