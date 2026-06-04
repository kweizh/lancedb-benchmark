import importlib
import os
import re
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

EXPECTED_QUERIES = [
    ("chromaspectrum", 7),
    ("hyperloomic", 23),
    ("zephyrglyph", 41),
]

MARK_OPEN = "<mark>"
MARK_CLOSE = "</mark>"
MARK_LEN = len(MARK_OPEN) + len(MARK_CLOSE)


def _load_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


def _open_table():
    import lancedb

    db = lancedb.connect(DATA_DIR)
    return db.open_table(os.environ["LANCE_TABLE"])


def _body_by_id(tbl, row_id):
    df = tbl.search().where(f"id = {int(row_id)}").limit(1).to_pandas()
    assert len(df) == 1, f"Could not load body for row id={row_id}"
    return df.iloc[0]["body"]


def _strip_marks(snippet):
    return snippet.replace(MARK_OPEN, "").replace(MARK_CLOSE, "")


def _mark_pair_re():
    return re.compile(re.escape(MARK_OPEN) + r"(.*?)" + re.escape(MARK_CLOSE), re.DOTALL)


def test_solution_module_importable():
    solution = _load_solution()
    assert hasattr(solution, "search_with_snippets"), (
        "solution.py must expose a callable named 'search_with_snippets'."
    )


@pytest.mark.parametrize("query,expected_top_id", EXPECTED_QUERIES)
def test_top1_matches_expected_id(query, expected_top_id):
    solution = _load_solution()
    results = solution.search_with_snippets(query, 3, 120)
    assert isinstance(results, list) and len(results) >= 1, (
        f"Query {query!r}: expected non-empty list, got {results!r}."
    )
    top = results[0]
    assert isinstance(top, dict), f"Top result for {query!r} must be a dict, got {type(top)}."
    assert int(top["id"]) == expected_top_id, (
        f"Query {query!r}: expected top-1 id={expected_top_id}, got id={top['id']}."
    )
    assert isinstance(top["score"], float) and top["score"] > 0.0, (
        f"Query {query!r}: expected positive float score, got {top['score']!r}."
    )


@pytest.mark.parametrize("query,expected_top_id", EXPECTED_QUERIES)
def test_snippet_length_within_budget(query, expected_top_id):
    solution = _load_solution()
    results = solution.search_with_snippets(query, 3, 120)
    for r in results:
        snippet = r["snippet"]
        # Length excluding markup must fit budget of 120 chars.
        stripped_len = len(_strip_marks(snippet))
        assert stripped_len <= 120, (
            f"Query {query!r}: snippet body length {stripped_len} exceeds snippet_chars=120."
        )
        # Total length with at most one <mark>...</mark> pair is bounded.
        assert len(snippet) <= 120 + MARK_LEN, (
            f"Query {query!r}: full snippet length {len(snippet)} exceeds 120+{MARK_LEN}."
        )


@pytest.mark.parametrize("query,expected_top_id", EXPECTED_QUERIES)
def test_mark_wraps_query_term(query, expected_top_id):
    solution = _load_solution()
    results = solution.search_with_snippets(query, 3, 120)
    top = results[0]
    snippet = top["snippet"]

    pair_re = _mark_pair_re()
    pairs = pair_re.findall(snippet)
    assert len(pairs) == 1, (
        f"Query {query!r}: snippet must contain exactly one <mark>...</mark> pair, got {len(pairs)}."
    )
    wrapped = pairs[0]
    # Wrapped text must equal the query term in a case-insensitive sense.
    assert wrapped.lower() == query.lower(), (
        f"Query {query!r}: <mark> wraps {wrapped!r}, expected case-insensitive match of {query!r}."
    )

    # Now look up the ground-truth body and confirm the case-preserved match.
    tbl = _open_table()
    body = _body_by_id(tbl, top["id"])
    match_pos = body.lower().find(query.lower())
    assert match_pos >= 0, (
        f"Query {query!r}: rigged body for id={top['id']} should contain the term."
    )
    expected_wrapped_in_body = body[match_pos : match_pos + len(query)]
    assert wrapped == expected_wrapped_in_body, (
        f"Query {query!r}: <mark> contents {wrapped!r} must equal body substring "
        f"{expected_wrapped_in_body!r} (case-preserved)."
    )


@pytest.mark.parametrize("query,expected_top_id", EXPECTED_QUERIES)
def test_snippet_offset_aligns_with_body(query, expected_top_id):
    solution = _load_solution()
    results = solution.search_with_snippets(query, 3, 120)
    top = results[0]
    snippet = top["snippet"]
    offset = int(top["snippet_offset"])

    tbl = _open_table()
    body = _body_by_id(tbl, top["id"])

    stripped = _strip_marks(snippet)
    window = body[offset : offset + len(stripped)]
    assert window == stripped, (
        f"Query {query!r}: body[{offset}:{offset+len(stripped)}] = {window!r}, "
        f"but stripped snippet = {stripped!r}."
    )

    match_pos = body.lower().find(query.lower())
    assert offset <= match_pos < offset + len(stripped), (
        f"Query {query!r}: match position {match_pos} not inside snippet window "
        f"[{offset}, {offset + len(stripped)})."
    )


def test_snippet_chars_clamp_is_respected():
    solution = _load_solution()
    results = solution.search_with_snippets("chromaspectrum", 5, 60)
    assert len(results) >= 1, "Expected at least one hit for 'chromaspectrum' with k=5."
    assert int(results[0]["id"]) == 7, (
        f"Top-1 id for 'chromaspectrum' should still be 7, got {results[0]['id']}."
    )
    for r in results:
        stripped_len = len(_strip_marks(r["snippet"]))
        assert stripped_len <= 60, (
            f"snippet body length {stripped_len} exceeds requested snippet_chars=60."
        )


def test_results_are_deterministic():
    solution = _load_solution()
    a = solution.search_with_snippets("chromaspectrum", 3, 120)
    b = solution.search_with_snippets("chromaspectrum", 3, 120)
    # Order, ids, snippets, offsets, and scores must match exactly across two consecutive runs.
    assert len(a) == len(b), "Two consecutive calls returned different result lengths."
    for ra, rb in zip(a, b):
        assert int(ra["id"]) == int(rb["id"]), (
            f"Non-deterministic ordering: ids differ {ra['id']} vs {rb['id']}."
        )
        assert ra["snippet"] == rb["snippet"], (
            f"Non-deterministic snippet for id={ra['id']}: {ra['snippet']!r} vs {rb['snippet']!r}."
        )
        assert int(ra["snippet_offset"]) == int(rb["snippet_offset"]), (
            f"Non-deterministic snippet_offset for id={ra['id']}."
        )
