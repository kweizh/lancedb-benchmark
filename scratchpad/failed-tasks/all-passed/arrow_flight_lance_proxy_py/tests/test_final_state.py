import json
import os
import socket
import sys
import time

import pytest
from xprocess import ProcessStarter

PROJECT_DIR = "/home/user/flight_proxy"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")
HOST = "127.0.0.1"
PORT = 8815


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


@pytest.fixture(scope="session")
def flight_server(xprocess):
    class Starter(ProcessStarter):
        name = "flight_server"
        args = [sys.executable, "server.py"]
        env = os.environ.copy()
        popen_kwargs = {
            "cwd": PROJECT_DIR,
            "text": True,
        }
        timeout = 60
        terminate_on_interrupt = True

        def startup_check(self):
            return _port_open(HOST, PORT)

    xprocess.ensure(Starter.name, Starter)

    # Extra readiness pause — let Flight's gRPC service finish wiring up.
    deadline = time.time() + 30
    while time.time() < deadline:
        if _port_open(HOST, PORT):
            break
        time.sleep(0.5)

    yield

    info = xprocess.getinfo(Starter.name)
    info.terminate()


@pytest.fixture(scope="session")
def lance_table():
    import lancedb

    conn = lancedb.connect(LANCEDB_DIR)
    return conn.open_table("documents")


@pytest.fixture(scope="session")
def flight_client(flight_server):
    import pyarrow.flight as flight

    client = flight.connect(f"grpc://{HOST}:{PORT}")
    # Wait until the server actually responds (a do_get call is cheaper
    # than list_flights to confirm the proxy is wired up). Retry briefly.
    yield client


def _query_vector_for(table, doc_id: str):
    # Read the full table (only 200 rows) and pick the requested row.
    # Avoids depending on `pylance` for `to_lance().to_table(filter=...)`.
    arrow_table = table.to_arrow()
    ids = arrow_table.column("id").to_pylist()
    assert doc_id in ids, f"Could not find row {doc_id} in documents table for ground truth."
    idx = ids.index(doc_id)
    return list(arrow_table.column("embedding")[idx].as_py())


def _flight_topk(client, vector, k):
    import pyarrow.flight as flight

    payload = json.dumps({"vector": list(map(float, vector)), "k": int(k)}).encode("utf-8")
    ticket = flight.Ticket(payload)
    reader = client.do_get(ticket)
    return reader.read_all()


def test_server_reachable(flight_client):
    assert _port_open(HOST, PORT), f"Flight server is not listening on {HOST}:{PORT}."


def test_topk_matches_ground_truth_for_doc_042(flight_client, lance_table):
    qvec = _query_vector_for(lance_table, "doc_042")

    direct = lance_table.search(qvec).limit(10).to_arrow()
    direct_ids = direct.column("id").to_pylist()
    assert len(direct_ids) == 10, "Direct LanceDB top-10 ground truth should have 10 ids."

    arrow_table = _flight_topk(flight_client, qvec, 10)

    assert arrow_table.num_rows == 10, (
        f"Flight response expected 10 rows, got {arrow_table.num_rows}."
    )

    field_names = set(arrow_table.schema.names)
    for col in ("id", "text", "embedding", "_distance"):
        assert col in field_names, (
            f"Flight schema missing required column '{col}'; got {field_names}."
        )

    flight_ids = arrow_table.column("id").to_pylist()
    assert flight_ids == direct_ids, (
        f"Flight top-10 ids {flight_ids} do not match LanceDB ground truth {direct_ids}."
    )
    assert flight_ids[0] == "doc_042", (
        f"Expected nearest neighbor of doc_042 to be itself, got {flight_ids[0]}."
    )

    distances = arrow_table.column("_distance").to_pylist()
    for prev, cur in zip(distances, distances[1:]):
        assert cur >= prev - 1e-6, (
            f"_distance column not monotonically non-decreasing: {distances}"
        )


def test_topk_respects_k_parameter(flight_client, lance_table):
    qvec = _query_vector_for(lance_table, "doc_042")
    direct_ids = lance_table.search(qvec).limit(10).to_arrow().column("id").to_pylist()

    arrow_table = _flight_topk(flight_client, qvec, 3)
    assert arrow_table.num_rows == 3, (
        f"Flight response with k=3 should have 3 rows; got {arrow_table.num_rows}."
    )
    flight_ids = arrow_table.column("id").to_pylist()
    assert flight_ids == direct_ids[:3], (
        f"Top-3 mismatch: flight={flight_ids}, expected={direct_ids[:3]}"
    )


def test_topk_matches_for_arbitrary_query_vector(flight_client, lance_table):
    qvec = _query_vector_for(lance_table, "doc_137")
    direct_ids = lance_table.search(qvec).limit(5).to_arrow().column("id").to_pylist()

    arrow_table = _flight_topk(flight_client, qvec, 5)
    flight_ids = arrow_table.column("id").to_pylist()
    assert flight_ids == direct_ids, (
        f"Top-5 mismatch for doc_137: flight={flight_ids}, expected={direct_ids}"
    )
    assert flight_ids[0] == "doc_137", (
        f"Expected nearest neighbor of doc_137 to be itself; got {flight_ids[0]}."
    )


def test_schema_preserves_fixed_size_list(flight_client, lance_table):
    import pyarrow as pa

    qvec = _query_vector_for(lance_table, "doc_042")
    arrow_table = _flight_topk(flight_client, qvec, 2)

    emb_type = arrow_table.schema.field("embedding").type
    assert pa.types.is_fixed_size_list(emb_type), (
        f"Expected embedding to be a fixed_size_list; got {emb_type}."
    )
    assert emb_type.list_size == 32, (
        f"Expected fixed_size_list of width 32; got width {emb_type.list_size}."
    )

    dist_type = arrow_table.schema.field("_distance").type
    assert pa.types.is_floating(dist_type), (
        f"Expected _distance to be a floating-point column; got {dist_type}."
    )
