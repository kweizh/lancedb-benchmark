import os
import socket
import time
import uuid

import pytest
import requests
from xprocess import ProcessStarter

PROJECT_DIR = "/home/user/myproject"
GRAPHQL_URL = "http://127.0.0.1:8000/graphql"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="session")
def graphql_server(xprocess):
    class Starter(ProcessStarter):
        name = "graphql_server"
        args = [
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        env = os.environ.copy()
        popen_kwargs = {"cwd": PROJECT_DIR, "text": True}
        timeout = 180
        terminate_on_interrupt = True
        max_read_lines = 2000

        def startup_check(self):
            if not _port_open("127.0.0.1", 8000):
                return False
            try:
                r = requests.post(
                    GRAPHQL_URL,
                    json={"query": "{ __typename }"},
                    timeout=5,
                )
                return r.status_code == 200
            except Exception:
                return False

    xprocess.ensure(Starter.name, Starter)
    yield
    info = xprocess.getinfo(Starter.name)
    info.terminate()


def _post(query: str, variables: dict | None = None) -> requests.Response:
    payload: dict = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    return requests.post(GRAPHQL_URL, json=payload, timeout=60)


def _read_seed_meta() -> dict:
    import json

    path = "/home/user/myproject/.seed_meta.json"
    assert os.path.isfile(path), (
        f"Expected seed metadata file {path} to be written by the entrypoint seeder."
    )
    with open(path) as f:
        return json.load(f)


def test_introspection_schema(graphql_server):
    r = _post("{ __schema { queryType { name } mutationType { name } } }")
    assert r.status_code == 200, f"Expected 200 for introspection, got {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors in introspection response: {body}"
    )
    data = body.get("data", {})
    schema = data.get("__schema") or {}
    qt = (schema.get("queryType") or {}).get("name")
    mt = (schema.get("mutationType") or {}).get("name")
    assert qt == "Query", f"Expected queryType name 'Query', got {qt!r}"
    assert mt == "Mutation", f"Expected mutationType name 'Mutation', got {mt!r}"


def test_vector_search_returns_seeded_doc(graphql_server):
    meta = _read_seed_meta()
    query_text = meta["row0_body"]
    q = (
        "query($q: String!, $k: Int!) { "
        "vectorSearch(query: $q, k: $k) { id score title snippet } }"
    )
    r = _post(q, {"q": query_text, "k": 5})
    assert r.status_code == 200, f"vectorSearch HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors from vectorSearch: {body}"
    )
    results = body["data"]["vectorSearch"]
    assert isinstance(results, list) and len(results) == 5, (
        f"Expected exactly 5 vectorSearch results, got: {results!r}"
    )
    ids = [r["id"] for r in results]
    assert 0 in ids, f"Expected seeded id=0 in vectorSearch top-5, got ids={ids}"
    for item in results:
        assert isinstance(item["score"], (int, float)), (
            f"DocResult.score must be a number, got: {item!r}"
        )
        assert len(item["snippet"]) <= 120, (
            f"DocResult.snippet must be <=120 chars, got len={len(item['snippet'])}"
        )
        assert isinstance(item["title"], str) and item["title"], (
            f"DocResult.title must be a non-empty string, got: {item!r}"
        )


def test_fts_search_returns_unique_sentinel_doc(graphql_server):
    meta = _read_seed_meta()
    sentinel = meta["row42_sentinel"]
    q = (
        "query($q: String!, $k: Int!) { "
        "ftsSearch(query: $q, k: $k) { id score title snippet } }"
    )
    r = _post(q, {"q": sentinel, "k": 5})
    assert r.status_code == 200, f"ftsSearch HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors from ftsSearch: {body}"
    )
    results = body["data"]["ftsSearch"]
    assert isinstance(results, list) and len(results) >= 1, (
        f"Expected at least 1 ftsSearch result, got: {results!r}"
    )
    ids = [r["id"] for r in results]
    assert ids[0] == 42, (
        f"Expected ftsSearch top hit to be id=42 (unique sentinel), got ids={ids}"
    )


def test_hybrid_search_returns_seeded_doc(graphql_server):
    meta = _read_seed_meta()
    query_text = meta["row0_body"]
    q = (
        "query($q: String!, $k: Int!) { "
        "hybridSearch(query: $q, k: $k) { id score title snippet } }"
    )
    r = _post(q, {"q": query_text, "k": 5})
    assert r.status_code == 200, f"hybridSearch HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors from hybridSearch: {body}"
    )
    results = body["data"]["hybridSearch"]
    assert isinstance(results, list) and len(results) >= 1, (
        f"Expected at least 1 hybridSearch result, got: {results!r}"
    )
    ids = [r["id"] for r in results]
    assert 0 in ids, f"Expected seeded id=0 in hybridSearch top results, got ids={ids}"


def test_filter_docs_tech_after_ts(graphql_server):
    q = (
        "query($t: String!, $ts: Int) { "
        "filterDocs(tag: $t, afterTs: $ts) { id tags publishedAt } }"
    )
    r = _post(q, {"t": "tech", "ts": 1700000000})
    assert r.status_code == 200, f"filterDocs HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors from filterDocs: {body}"
    )
    rows = body["data"]["filterDocs"]
    assert isinstance(rows, list) and len(rows) >= 1, (
        f"Expected at least one tech row with publishedAt>=1700000000, got: {rows!r}"
    )
    for row in rows:
        assert "tech" in row["tags"], (
            f"filterDocs returned a row without 'tech' tag: {row!r}"
        )
        assert row["publishedAt"] >= 1700000000, (
            f"filterDocs returned a row with publishedAt<1700000000: {row!r}"
        )


def test_filter_docs_science_without_after_ts(graphql_server):
    q = "query($t: String!) { filterDocs(tag: $t) { id tags } }"
    r = _post(q, {"t": "science"})
    assert r.status_code == 200, f"filterDocs HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors from filterDocs: {body}"
    )
    rows = body["data"]["filterDocs"]
    assert isinstance(rows, list) and len(rows) >= 1, (
        f"Expected at least one 'science' row, got: {rows!r}"
    )
    for row in rows:
        assert "science" in row["tags"], (
            f"filterDocs returned a row without 'science' tag: {row!r}"
        )


def test_mutation_then_vector_search_finds_new_doc(graphql_server):
    marker = "MUT_" + uuid.uuid4().hex[:10].upper()
    new_body = f"This is a fresh mutation document with marker {marker} and additional content for embedding."
    new_title = f"new-doc-{marker.lower()}"

    mutation = (
        "mutation($t: String!, $b: String!) { "
        "addDocument(title: $t, body: $b) { id title body publishedAt tags } }"
    )
    r = _post(mutation, {"t": new_title, "b": new_body})
    assert r.status_code == 200, f"addDocument HTTP {r.status_code}: {r.text}"
    body = r.json()
    assert "errors" not in body or not body["errors"], (
        f"Unexpected errors from addDocument: {body}"
    )
    new_doc = body["data"]["addDocument"]
    assert new_doc["title"] == new_title, (
        f"addDocument returned wrong title: {new_doc!r}"
    )
    assert new_doc["body"] == new_body, (
        f"addDocument returned wrong body: {new_doc!r}"
    )
    assert isinstance(new_doc["id"], int) and new_doc["id"] >= 300, (
        f"addDocument id must be int >= 300 (seeded rows are 0..299), got: {new_doc!r}"
    )
    assert new_doc["tags"] == [], f"addDocument default tags must be []; got: {new_doc!r}"
    assert isinstance(new_doc["publishedAt"], int) and new_doc["publishedAt"] > 0, (
        f"addDocument publishedAt must be a positive int, got: {new_doc!r}"
    )

    new_id = new_doc["id"]

    # Allow indexing/flush to settle
    time.sleep(1.0)

    follow = (
        "query($q: String!, $k: Int!) { "
        "vectorSearch(query: $q, k: $k) { id title } }"
    )
    r2 = _post(follow, {"q": new_body, "k": 5})
    assert r2.status_code == 200, f"follow-up vectorSearch HTTP {r2.status_code}: {r2.text}"
    body2 = r2.json()
    assert "errors" not in body2 or not body2["errors"], (
        f"Unexpected errors from follow-up vectorSearch: {body2}"
    )
    results = body2["data"]["vectorSearch"]
    ids = [item["id"] for item in results]
    assert new_id in ids, (
        f"Expected the newly inserted id {new_id} to appear in vectorSearch top-5 "
        f"after mutation, got ids={ids}"
    )


def test_invalid_input_returns_graphql_errors(graphql_server):
    q = "query { vectorSearch(query: \"anything\", k: 0) { id } }"
    r = _post(q)
    assert r.status_code == 200, (
        f"k=0 must surface as a GraphQL-level error with HTTP 200, got {r.status_code}: {r.text}"
    )
    body = r.json()
    errors = body.get("errors")
    assert errors and isinstance(errors, list) and len(errors) >= 1, (
        f"Expected non-empty 'errors' array in response for k=0, got: {body!r}"
    )
