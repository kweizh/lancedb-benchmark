import json
import sqlite3
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
CONFIG_PATH = "/app/data/config.json"
QUERIES_PATH = "/app/data/queries.json"

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _load_queries():
    with open(QUERIES_PATH) as f:
        return json.load(f)


def _canonical_query():
    queries = _load_queries()
    q0 = queries[0]
    return q0["query"], [float(x) for x in q0["vector"]]


def _bm25_order(term, title_weight, body_weight, limit):
    cfg = _load_config()
    con = sqlite3.connect(cfg["sqlite_path"])
    try:
        table = cfg["table_name"]
        sql = (
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? "
            f"ORDER BY bm25({table}, {float(title_weight)}, {float(body_weight)}), rowid "
            f"LIMIT {int(limit)}"
        )
        rows = con.execute(sql, (term,)).fetchall()
    finally:
        con.close()
    return [int(r[0]) for r in rows]


def _vector_order(query_vector, limit):
    import lancedb

    cfg = _load_config()
    db = lancedb.connect(cfg["lancedb_uri"])
    tbl = db.open_table(cfg["table_name"])
    res = tbl.search(query_vector).distance_type(cfg["distance_metric"]).limit(int(limit)).to_list()
    res.sort(key=lambda r: (r["_distance"], int(r["id"])))
    return [int(r["id"]) for r in res]


def _weighted_rrf(bm25_ids, vector_ids, k, keyword_weight, rrf_k):
    bm25_rank = {did: i + 1 for i, did in enumerate(bm25_ids)}
    vec_rank = {did: i + 1 for i, did in enumerate(vector_ids)}
    all_ids = set(bm25_ids) | set(vector_ids)
    scored = []
    for did in all_ids:
        s = 0.0
        if did in bm25_rank:
            s += keyword_weight * (1.0 / (rrf_k + bm25_rank[did]))
        if did in vec_rank:
            s += (1.0 - keyword_weight) * (1.0 / (rrf_k + vec_rank[did]))
        scored.append((did, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [did for did, _ in scored[:k]]


def _ids(results):
    return [int(r["id"]) for r in results]


def test_solution_importable():
    import solution

    assert hasattr(solution, "search"), "solution.py must expose a callable `search`."
    assert callable(solution.search), "`solution.search` must be callable."


def test_pure_keyword_reproduces_bm25_order():
    import solution

    term, v0 = _canonical_query()
    cfg = _load_config()
    expected = _bm25_order(term, 1.0, 1.0, 5)
    assert len(expected) >= 5, "Fixture must have at least 5 keyword matches for this test."
    got = _ids(solution.search(term, v0, 5, 1.0))
    assert got == expected, (
        f"keyword_weight=1.0 must reproduce pure BM25 top-5. Expected {expected}, got {got}."
    )


def test_pure_vector_reproduces_vector_order():
    import solution

    term, v0 = _canonical_query()
    expected = _vector_order(v0, 5)
    got = _ids(solution.search(term, v0, 5, 0.0))
    assert got == expected, (
        f"keyword_weight=0.0 must reproduce pure vector top-5. Expected {expected}, got {got}."
    )


def test_intermediate_weight_surfaces_middle_doc():
    import solution

    term, v0 = _canonical_query()
    cfg = _load_config()
    top_kw = _ids(solution.search(term, v0, 5, 1.0))[0]
    top_vec = _ids(solution.search(term, v0, 5, 0.0))[0]
    top_mid = _ids(solution.search(term, v0, 5, 0.5))[0]
    assert len({top_kw, top_vec, top_mid}) == 3, (
        "The keyword-extreme, vector-extreme, and intermediate-blend winners must be three "
        f"distinct documents; got kw={top_kw}, vec={top_vec}, mid={top_mid}."
    )
    bm25_ids = _bm25_order(term, 1.0, 1.0, 50)
    vector_ids = _vector_order(v0, 50)
    expected_mid = _weighted_rrf(bm25_ids, vector_ids, 5, 0.5, cfg["rrf_k"])
    got_mid = _ids(solution.search(term, v0, 5, 0.5))
    assert got_mid == expected_mid, (
        f"keyword_weight=0.5 fused order must match weighted RRF. Expected {expected_mid}, got {got_mid}."
    )


def test_bm25_column_weighting_changes_ranking():
    import solution

    term, v0 = _canonical_query()

    title_weighted = _bm25_order(term, 8.0, 1.0, 8)
    body_weighted = _bm25_order(term, 1.0, 8.0, 8)
    assert title_weighted != body_weighted, (
        "Fixture invariant broken: title-weighted and body-weighted BM25 orders should differ."
    )

    got_a = _ids(solution.search(term, v0, 8, 1.0, title_weight=8.0, body_weight=1.0))
    got_b = _ids(solution.search(term, v0, 8, 1.0, title_weight=1.0, body_weight=8.0))
    assert got_a == title_weighted, (
        f"Title-weighted search must match title-weighted BM25 order. Expected {title_weighted}, got {got_a}."
    )
    assert got_b == body_weighted, (
        f"Body-weighted search must match body-weighted BM25 order. Expected {body_weighted}, got {got_b}."
    )

    # The document whose only 'photon' hit is in its title is unique in the fixture.
    # It must rank strictly higher when the title column is up-weighted than when
    # the body column is up-weighted.
    cfg = _load_config()
    con = sqlite3.connect(cfg["sqlite_path"])
    try:
        rows = con.execute(f"SELECT rowid, title, body FROM {cfg['table_name']}").fetchall()
    finally:
        con.close()
    title_only = [int(r[0]) for r in rows if "photon" in r[1].lower() and "photon" not in r[2].lower()]
    assert len(title_only) == 1, (
        f"Fixture must contain exactly one title-only photon doc; got {title_only}."
    )
    t_doc = title_only[0]
    assert t_doc in got_a and t_doc in got_b, (
        f"The title-only photon doc {t_doc} must appear in both weighted result sets."
    )
    assert got_a.index(t_doc) < got_b.index(t_doc), (
        "Up-weighting the title column must lift the title-only photon doc to a higher rank "
        f"than up-weighting the body column (title-weighted idx {got_a.index(t_doc)} vs "
        f"body-weighted idx {got_b.index(t_doc)})."
    )


def test_return_shape_and_truncation():
    import solution

    term, v0 = _canonical_query()
    results = solution.search(term, v0, 3, 0.5)
    assert isinstance(results, list) and len(results) == 3, "search must return a list of length k=3."
    prev = None
    for item in results:
        assert isinstance(item, dict), "Each result must be a dict."
        assert set(item.keys()) == {"id", "score"}, (
            f"Each result must have exactly keys 'id' and 'score'; got {set(item.keys())}."
        )
        assert isinstance(item["id"], int), "Result 'id' must be an int."
        assert isinstance(item["score"], float), "Result 'score' must be a float."
        if prev is not None:
            assert item["score"] <= prev + 1e-12, "Scores must be non-increasing."
        prev = item["score"]


def test_determinism():
    import solution

    term, v0 = _canonical_query()
    first = solution.search(term, v0, 5, 0.5)
    second = solution.search(term, v0, 5, 0.5)
    assert first == second, "search must be deterministic across identical calls."
