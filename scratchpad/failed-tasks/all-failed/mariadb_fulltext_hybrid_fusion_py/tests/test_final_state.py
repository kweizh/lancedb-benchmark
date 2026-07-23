import importlib
import json
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
EXPECTED_JSON = "/app/expected.json"
RRF_K = 60


@pytest.fixture(scope="session")
def expected():
    with open(EXPECTED_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    mod = importlib.import_module("solution")
    assert hasattr(mod, "hybrid_search"), "solution.py must expose a hybrid_search function."
    return mod


def _run_id():
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id and os.path.isfile("/logs/artifacts/run-id"):
        with open("/logs/artifacts/run-id") as f:
            run_id = f.read().strip()
    assert run_id, "ZEALT_RUN_ID environment variable is not set."
    return run_id


def _table_name():
    return f"docs_{_run_id()}"


def _mariadb_conn():
    import pymysql

    return pymysql.connect(
        host=os.environ.get("MARIADB_HOST", "127.0.0.1"),
        port=int(os.environ.get("MARIADB_PORT", "3306")),
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        database=os.environ["MARIADB_DATABASE"],
    )


def _fts_ranked_ids(query_text):
    """Independent FULLTEXT ranking (best match first)."""
    table = _table_name()
    conn = _mariadb_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) AS rel "
                f"FROM `{table}` "
                f"WHERE MATCH(body) AGAINST (%s IN NATURAL LANGUAGE MODE) "
                f"ORDER BY rel DESC, id ASC",
                (query_text, query_text),
            )
            return [int(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _vector_ranked_ids(query_vector, limit=50):
    """Independent LanceDB cosine ranking (nearest first)."""
    import lancedb

    db = lancedb.connect(os.environ["LANCEDB_PATH"])
    tbl = db.open_table(_table_name())
    rows = tbl.search(query_vector).distance_type("cosine").limit(limit).to_list()
    return [int(r["id"]) for r in rows]


def test_return_contract(solution, expected):
    result = solution.hybrid_search(expected["query_text"], expected["query_vector"], 5)
    assert isinstance(result, list), "hybrid_search must return a list."
    assert 1 <= len(result) <= 5, f"Expected between 1 and 5 results, got {len(result)}."
    for item in result:
        assert isinstance(item, dict), f"Each result must be a dict, got {type(item)}."
        assert set(item.keys()) == {"id", "score"}, \
            f"Each result must have exactly keys id and score, got {sorted(item.keys())}."
        assert isinstance(item["id"], int), f"id must be an int, got {type(item['id'])}."
        assert isinstance(item["score"], float), f"score must be a float, got {type(item['score'])}."
    ordering_key = [(-item["score"], item["id"]) for item in result]
    assert ordering_key == sorted(ordering_key), \
        "Results must be sorted by score descending, ties broken by id ascending."


def test_both_list_document_wins(solution, expected):
    result = solution.hybrid_search(expected["query_text"], expected["query_vector"], 5)
    assert result, "hybrid_search returned an empty list."
    assert result[0]["id"] == expected["both_id"], (
        f"Expected the both-list document {expected['both_id']} at rank 1, "
        f"got {result[0]['id']}."
    )
    top_score = result[0]["score"]
    for item in result[1:]:
        assert top_score > item["score"], (
            f"The both-list document must have a strictly higher fused score than "
            f"document {item['id']} (top={top_score}, other={item['score']})."
        )


def test_single_list_documents_included(solution, expected):
    result = solution.hybrid_search(expected["query_text"], expected["query_vector"], 5)
    ids = {item["id"] for item in result}
    assert expected["keyword_only_id"] in ids, (
        f"keyword-only document {expected['keyword_only_id']} must appear in the fused top-k."
    )
    assert expected["vector_only_id"] in ids, (
        f"vector-only document {expected['vector_only_id']} must appear in the fused top-k."
    )


def test_fusion_combines_both_engines(expected):
    fts_ids = _fts_ranked_ids(expected["query_text"])
    assert expected["keyword_only_id"] in fts_ids, (
        "keyword-only document should be returned by the FULLTEXT engine."
    )
    assert expected["vector_only_id"] not in fts_ids, (
        "vector-only document must NOT be returned by the FULLTEXT engine."
    )

    vec_ids = _vector_ranked_ids(expected["query_vector"], limit=50)
    top_vec = vec_ids[:3]
    assert expected["vector_only_id"] in top_vec, (
        f"vector-only document {expected['vector_only_id']} should rank near the top "
        f"of the vector search, top-3 were {top_vec}."
    )
    assert expected["keyword_only_id"] not in vec_ids[:10], (
        f"keyword-only document {expected['keyword_only_id']} should be far down the "
        f"vector ranking (not in top-10), top-10 were {vec_ids[:10]}."
    )


def test_rrf_score_matches_formula(solution, expected):
    fts_ids = _fts_ranked_ids(expected["query_text"])
    vec_ids = _vector_ranked_ids(expected["query_vector"], limit=1000)
    both_id = expected["both_id"]

    r_fts = fts_ids.index(both_id) + 1
    r_vec = vec_ids.index(both_id) + 1
    expected_score = 1.0 / (RRF_K + r_fts) + 1.0 / (RRF_K + r_vec)

    result = solution.hybrid_search(expected["query_text"], expected["query_vector"], 5)
    got = {item["id"]: item["score"] for item in result}
    assert both_id in got, f"both_id {both_id} missing from results."
    assert abs(got[both_id] - expected_score) < 1e-6, (
        f"Fused score for {both_id} should match the RRF formula "
        f"(expected {expected_score}, got {got[both_id]})."
    )


def test_determinism(solution, expected):
    first = solution.hybrid_search(expected["query_text"], expected["query_vector"], 5)
    second = solution.hybrid_search(expected["query_text"], expected["query_vector"], 5)
    assert first == second, (
        f"hybrid_search must be deterministic; got {first} then {second}."
    )
