import importlib
import json
import os

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_FIXTURE = os.path.join(PROJECT_DIR, ".expected.json")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), (
        f"Seeded LanceDB directory {LANCEDB_DIR} does not exist; "
        "the build-time seed step did not run."
    )


def test_lancedb_importable():
    importlib.import_module("lancedb")


def test_documents_table_seeded():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    assert "documents" in db.table_names(), (
        "LanceDB table 'documents' was not seeded; expected at /home/user/myproject/data/lancedb."
    )
    tbl = db.open_table("documents")
    n = tbl.count_rows()
    assert n == 500, f"Expected 500 seeded rows in 'documents', got {n}."


def test_documents_table_schema():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("documents")
    field_names = {f.name for f in tbl.schema}
    for name in ("id", "title", "category", "vector"):
        assert name in field_names, (
            f"Field '{name}' missing from 'documents' table schema; got {sorted(field_names)}."
        )


def test_expected_fixture_present_and_well_formed():
    assert os.path.isfile(EXPECTED_FIXTURE), (
        f"Build-time fixture {EXPECTED_FIXTURE} is missing; "
        "the verifier needs it to know the expected retrieval result."
    )
    with open(EXPECTED_FIXTURE) as f:
        payload = json.load(f)
    assert isinstance(payload, dict), "Fixture must be a JSON object."

    qv = payload.get("query_vector")
    assert isinstance(qv, list) and len(qv) == 24 and all(isinstance(x, (int, float)) for x in qv), (
        "Fixture must contain a 'query_vector' list of 24 numeric values."
    )

    for key in ("top_k_no_filter", "top_k_with_filter"):
        hits = payload.get(key)
        assert isinstance(hits, list) and len(hits) == 5, (
            f"Fixture key '{key}' must be a list of length 5."
        )
        for h in hits:
            assert isinstance(h, dict) and isinstance(h.get("id"), int) and isinstance(h.get("title"), str), (
                f"Each hit under '{key}' must have integer 'id' and string 'title'."
            )

    wc = payload.get("where_clause")
    assert isinstance(wc, str) and wc.strip(), "Fixture must contain a non-empty 'where_clause' string."


def test_grpc_libs_importable():
    importlib.import_module("grpc")
    importlib.import_module("grpc_tools.protoc")


def test_verifier_deps_importable():
    importlib.import_module("xprocess")


def test_no_candidate_artifacts_yet():
    # The candidate is expected to generate these; they MUST NOT be pre-populated.
    for fname in ("server.py", "search_pb2.py", "search_pb2_grpc.py", "proto/search.proto"):
        assert not os.path.exists(os.path.join(PROJECT_DIR, fname)), (
            f"Unexpected pre-existing artifact at {os.path.join(PROJECT_DIR, fname)}; the candidate should create it."
        )
