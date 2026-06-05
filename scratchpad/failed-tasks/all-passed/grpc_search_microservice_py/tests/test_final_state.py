import importlib
import json
import math
import os
import socket
import sys

import pytest
from xprocess import ProcessStarter

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_FIXTURE = os.path.join(PROJECT_DIR, ".expected.json")

HOST = "127.0.0.1"
PORT = 50051
TARGET = f"{HOST}:{PORT}"


def _load_fixture():
    with open(EXPECTED_FIXTURE) as f:
        return json.load(f)


def _import_stubs():
    """Import candidate-generated gRPC stubs from the project root.

    Inserts the project root in sys.path so that `import search_pb2` and
    `import search_pb2_grpc` resolve to the candidate's generated modules.
    """
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Drop any cached modules from a previous import attempt so that we
    # always pick up the candidate-generated files.
    for mod_name in ("search_pb2", "search_pb2_grpc"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    pb2 = importlib.import_module("search_pb2")
    pb2_grpc = importlib.import_module("search_pb2_grpc")
    return pb2, pb2_grpc


@pytest.fixture(scope="session")
def start_server(xprocess):
    class Starter(ProcessStarter):
        name = "grpc_search_server"
        args = ["python3", "server.py"]
        env = os.environ.copy()
        popen_kwargs = {
            "cwd": PROJECT_DIR,
            "text": True,
        }
        timeout = 120
        terminate_on_interrupt = True

        def startup_check(self):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((HOST, PORT)) == 0

    xprocess.ensure(Starter.name, Starter)
    yield
    info = xprocess.getinfo(Starter.name)
    info.terminate()


def test_generated_stub_files_exist():
    for fname in ("search_pb2.py", "search_pb2_grpc.py"):
        path = os.path.join(PROJECT_DIR, fname)
        assert os.path.isfile(path), (
            f"Expected generated gRPC stub at {path}. "
            "Run `python3 -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/search.proto` "
            "from the project root."
        )


def test_proto_contract_field_shapes():
    pb2, pb2_grpc = _import_stubs()

    # SearchRequest
    sr_fields = {f.name: f for f in pb2.SearchRequest.DESCRIPTOR.fields}
    for name in ("vector", "k", "where_clause"):
        assert name in sr_fields, (
            f"SearchRequest is missing field '{name}'. Found: {sorted(sr_fields)}"
        )
    # vector: repeated float
    v = sr_fields["vector"]
    assert v.label == v.LABEL_REPEATED, "SearchRequest.vector must be 'repeated'."
    assert v.type == v.TYPE_FLOAT, "SearchRequest.vector must be type 'float'."
    # k: int32
    k = sr_fields["k"]
    assert k.type == k.TYPE_INT32, "SearchRequest.k must be type 'int32'."
    # where_clause: string
    wc = sr_fields["where_clause"]
    assert wc.type == wc.TYPE_STRING, "SearchRequest.where_clause must be type 'string'."

    # SearchResponse
    resp_fields = {f.name: f for f in pb2.SearchResponse.DESCRIPTOR.fields}
    assert "hits" in resp_fields, "SearchResponse must have a 'hits' field."
    hits_f = resp_fields["hits"]
    assert hits_f.label == hits_f.LABEL_REPEATED, "SearchResponse.hits must be 'repeated'."
    assert hits_f.type == hits_f.TYPE_MESSAGE, "SearchResponse.hits must be a message type."
    assert hits_f.message_type.name == "Hit", (
        f"SearchResponse.hits message type must be 'Hit', got {hits_f.message_type.name}"
    )

    # Hit
    hit_fields = {f.name: f for f in pb2.Hit.DESCRIPTOR.fields}
    for name in ("id", "score", "title"):
        assert name in hit_fields, f"Hit is missing field '{name}'. Found: {sorted(hit_fields)}"
    assert hit_fields["id"].type == hit_fields["id"].TYPE_INT64, "Hit.id must be type 'int64'."
    assert hit_fields["score"].type == hit_fields["score"].TYPE_FLOAT, "Hit.score must be type 'float'."
    assert hit_fields["title"].type == hit_fields["title"].TYPE_STRING, "Hit.title must be type 'string'."

    # Stub exposed
    assert hasattr(pb2_grpc, "SearchServiceStub"), (
        "search_pb2_grpc must expose 'SearchServiceStub'."
    )


def test_unfiltered_search_matches_ground_truth(start_server):
    import grpc

    pb2, pb2_grpc = _import_stubs()
    fixture = _load_fixture()
    query_vector = [float(x) for x in fixture["query_vector"]]
    expected = fixture["top_k_no_filter"]

    with grpc.insecure_channel(TARGET) as channel:
        stub = pb2_grpc.SearchServiceStub(channel)
        req = pb2.SearchRequest(vector=query_vector, k=5, where_clause="")
        resp = stub.Search(req, timeout=30)

    assert len(resp.hits) == 5, f"Expected 5 hits, got {len(resp.hits)}"

    got_ids = [int(h.id) for h in resp.hits]
    exp_ids = [int(h["id"]) for h in expected]
    assert got_ids == exp_ids, (
        f"Unfiltered top-5 IDs do not match ground truth.\n"
        f"  expected: {exp_ids}\n  got:      {got_ids}"
    )

    got_titles = [h.title for h in resp.hits]
    exp_titles = [h["title"] for h in expected]
    assert got_titles == exp_titles, (
        f"Unfiltered top-5 titles do not match ground truth.\n"
        f"  expected: {exp_titles}\n  got:      {got_titles}"
    )

    scores = [float(h.score) for h in resp.hits]
    assert all(math.isfinite(s) and s >= 0 for s in scores), (
        f"All Hit.score values must be finite and non-negative, got {scores}"
    )
    assert len(set(scores)) > 1, (
        f"Hit.score values are all identical ({scores[0]}); they should reflect real LanceDB distances."
    )


def test_filtered_search_enforced_server_side(start_server):
    import grpc
    import lancedb

    pb2, pb2_grpc = _import_stubs()
    fixture = _load_fixture()
    query_vector = [float(x) for x in fixture["query_vector"]]
    where_clause = fixture["where_clause"]
    expected = fixture["top_k_with_filter"]

    with grpc.insecure_channel(TARGET) as channel:
        stub = pb2_grpc.SearchServiceStub(channel)
        req = pb2.SearchRequest(vector=query_vector, k=5, where_clause=where_clause)
        resp = stub.Search(req, timeout=30)

    assert len(resp.hits) == 5, f"Expected 5 hits, got {len(resp.hits)}"

    got_ids = [int(h.id) for h in resp.hits]
    exp_ids = [int(h["id"]) for h in expected]
    assert got_ids == exp_ids, (
        f"Filtered top-5 IDs do not match ground truth.\n"
        f"  expected: {exp_ids}\n  got:      {got_ids}"
    )

    got_titles = [h.title for h in resp.hits]
    exp_titles = [h["title"] for h in expected]
    assert got_titles == exp_titles, (
        f"Filtered top-5 titles do not match ground truth.\n"
        f"  expected: {exp_titles}\n  got:      {got_titles}"
    )

    # Cross-check that every returned id actually has category == 'alpha' in LanceDB.
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("documents")
    id_list = ",".join(str(i) for i in got_ids)
    rows = tbl.search().where(f"id IN ({id_list})").limit(len(got_ids)).to_list()
    assert len(rows) == len(got_ids), (
        f"LanceDB returned {len(rows)} rows for filter `id IN ({id_list})`, expected {len(got_ids)}"
    )
    bad = [r for r in rows if r.get("category") != "alpha"]
    assert not bad, (
        f"Server returned hits whose category != 'alpha' (where_clause was not enforced): {bad}"
    )
