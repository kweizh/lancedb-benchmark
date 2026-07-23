import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time

import numpy as np
import pika
import pytest

PROJECT_DIR = "/home/user/project"
DB_PATH = "/home/user/project/data/lancedb"
COMMITS_LOG = "/home/user/project/data/commits.log"
TABLE_NAME = "documents"

MAIN_QUEUE = "documents"
DLX = "documents.dlx"
DLQ = "documents.dlq"

HOST = "127.0.0.1"
PORT = 5672

MAIN_QUEUE_ARGS = {
    "x-queue-type": "quorum",
    "x-dead-letter-exchange": DLX,
}
DLQ_ARGS = {"x-queue-type": "quorum"}

EMBED_DIM = 64


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0


def get_connection(retries: int = 30, delay: float = 2.0) -> pika.BlockingConnection:
    params = pika.ConnectionParameters(
        host=HOST,
        port=PORT,
        virtual_host="/",
        credentials=pika.PlainCredentials("guest", "guest"),
        heartbeat=0,
        blocked_connection_timeout=30,
    )
    last_err = None
    for _ in range(retries):
        try:
            return pika.BlockingConnection(params)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to RabbitMQ: {last_err}")


def declare_topology() -> None:
    conn = get_connection()
    ch = conn.channel()
    ch.exchange_declare(exchange=DLX, exchange_type="fanout", durable=True)
    ch.queue_declare(queue=DLQ, durable=True, arguments=DLQ_ARGS)
    ch.queue_bind(queue=DLQ, exchange=DLX)
    ch.queue_declare(queue=MAIN_QUEUE, durable=True, arguments=MAIN_QUEUE_ARGS)
    conn.close()


def _delete(name: str, is_exchange: bool = False) -> None:
    # Use a fresh connection per op so a broker-side channel close does not
    # cascade into the next delete.
    try:
        conn = get_connection()
        ch = conn.channel()
        if is_exchange:
            ch.exchange_delete(exchange=name)
        else:
            ch.queue_delete(queue=name)
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"(cleanup) ignoring delete error for {name}: {exc}")


def clean_state() -> None:
    _delete(MAIN_QUEUE)
    _delete(DLQ)
    _delete(DLX, is_exchange=True)
    shutil.rmtree(DB_PATH, ignore_errors=True)
    if os.path.isfile(COMMITS_LOG):
        os.remove(COMMITS_LOG)


def publish_messages(messages) -> None:
    """messages: list of (body_bytes, message_id_or_None)."""
    conn = get_connection()
    ch = conn.channel()
    ch.confirm_delivery()
    for body, msg_id in messages:
        props = pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
            message_id=msg_id,
        )
        ch.basic_publish(
            exchange="",
            routing_key=MAIN_QUEUE,
            body=body,
            properties=props,
            mandatory=True,
        )
    conn.close()


def queue_message_count(qname: str) -> int:
    conn = get_connection()
    ch = conn.channel()
    method = ch.queue_declare(queue=qname, durable=True, passive=True)
    count = method.method.message_count
    conn.close()
    return count


def wait_for_count(qname: str, expected: int, timeout: float = 30.0) -> int:
    deadline = time.time() + timeout
    count = -1
    while time.time() < deadline:
        try:
            count = queue_message_count(qname)
        except Exception as exc:  # noqa: BLE001
            print(f"(wait) count error for {qname}: {exc}")
            count = -1
        if count == expected:
            return count
        time.sleep(1.0)
    return count


def run_pipeline() -> str:
    result = subprocess.run(
        ["python3", "ingest.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("----- ingest.py stdout -----")
    print(result.stdout)
    print("----- ingest.py stderr -----")
    print(result.stderr)
    assert result.returncode == 0, (
        f"ingest.py exited with code {result.returncode}. stderr: {result.stderr}"
    )
    return result.stdout


def parse_summary(stdout: str):
    m = re.search(
        r"INGEST_DONE\s+written=(\d+)\s+skipped_duplicates=(\d+)\s+dead_lettered=(\d+)",
        stdout,
    )
    assert m is not None, (
        "Expected a summary line matching 'INGEST_DONE written=<W> "
        f"skipped_duplicates=<D> dead_lettered=<P>' in stdout, got:\n{stdout}"
    )
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def embed(text: str) -> np.ndarray:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vec = np.zeros(EMBED_DIM, dtype=np.float64)
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % EMBED_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def load_table_rows():
    import lancedb

    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    rows = {}
    for _, row in df.iterrows():
        rows[row["id"]] = row
    return db, tbl, rows


def doc_text(i: int) -> str:
    return f"alpha bravo charlie document {i} unique{i}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def rabbitmq_broker():
    if not _port_open(HOST, PORT):
        proc = subprocess.run(
            ["rabbitmq-server", "-detached"],
            capture_output=True,
            text=True,
        )
        print("rabbitmq-server -detached stdout:", proc.stdout)
        print("rabbitmq-server -detached stderr:", proc.stderr)

    # Best-effort blocking wait for the node to finish booting.
    subprocess.run(
        ["rabbitmqctl", "await_startup"],
        capture_output=True,
        text=True,
        timeout=180,
    )

    deadline = time.time() + 180
    while time.time() < deadline:
        if _port_open(HOST, PORT):
            break
        time.sleep(2)
    assert _port_open(HOST, PORT), (
        f"RabbitMQ broker is not reachable on {HOST}:{PORT}."
    )
    # Confirm an actual AMQP handshake succeeds before proceeding.
    get_connection().close()
    yield


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
def test_full_ingestion_flow():
    # ---- Setup: start from a clean, test-controlled state --------------- #
    clean_state()
    declare_topology()

    # =================================================================== #
    # Run 1 — dedup within a run + poison dead-lettering
    # =================================================================== #
    run1 = []
    # 20 valid unique docs doc-00 .. doc-19
    for i in range(20):
        did = f"doc-{i:02d}"
        body = json.dumps({"id": did, "text": doc_text(i)}).encode("utf-8")
        run1.append((body, did))
    # 5 in-run duplicates: doc-00 .. doc-04 again
    for i in range(5):
        did = f"doc-{i:02d}"
        body = json.dumps({"id": did, "text": doc_text(i)}).encode("utf-8")
        run1.append((body, did))
    # 3 poison messages
    run1.append((b"{not valid json", None))
    run1.append((b"[1, 2, 3]", None))
    run1.append((json.dumps({"id": "bad-1"}).encode("utf-8"), "bad-1"))

    publish_messages(run1)
    stdout1 = run_pipeline()

    written, dupes, dead = parse_summary(stdout1)
    assert (written, dupes, dead) == (20, 5, 3), (
        "Run 1 summary mismatch. Expected written=20 skipped_duplicates=5 "
        f"dead_lettered=3, got written={written} skipped_duplicates={dupes} "
        f"dead_lettered={dead}."
    )

    # LanceDB table content
    _, tbl, rows = load_table_rows()
    expected_ids_1 = {f"doc-{i:02d}" for i in range(20)}
    assert len(rows) == 20, f"Expected 20 rows after run 1, got {len(rows)}."
    assert set(rows.keys()) == expected_ids_1, (
        f"Run 1 id set mismatch. Missing={expected_ids_1 - set(rows.keys())}, "
        f"extra={set(rows.keys()) - expected_ids_1}."
    )

    # Deterministic embedding check for doc-07
    stored = np.asarray(rows["doc-07"]["vector"], dtype=np.float64)
    expected_vec = embed(doc_text(7))
    assert stored.shape[0] == EMBED_DIM, (
        f"Stored vector for doc-07 must have {EMBED_DIM} dims, got {stored.shape[0]}."
    )
    assert np.allclose(stored, expected_vec, atol=1e-5), (
        "Stored embedding for doc-07 does not match the deterministic embedding."
    )
    assert abs(np.linalg.norm(stored) - 1.0) < 1e-4, (
        "Stored embedding for doc-07 is not L2-normalized."
    )

    # Vector search returns doc-07 for its own embedding
    search_df = tbl.search(expected_vec.tolist()).limit(1).to_pandas()
    assert len(search_df) >= 1, "Vector search returned no results for doc-07."
    top_id = search_df.iloc[0]["id"]
    assert top_id == "doc-07", (
        f"Nearest neighbor of doc-07's embedding should be doc-07, got {top_id}."
    )

    # DLQ and main queue state
    dlq_count = wait_for_count(DLQ, 3)
    assert dlq_count == 3, (
        f"Expected 3 dead-lettered messages in {DLQ} after run 1, got {dlq_count}."
    )
    main_count = wait_for_count(MAIN_QUEUE, 0)
    assert main_count == 0, (
        f"Expected main queue {MAIN_QUEUE} to be empty after run 1, got {main_count}."
    )

    # Commit log structure
    assert os.path.isfile(COMMITS_LOG), f"Commit log {COMMITS_LOG} does not exist."
    with open(COMMITS_LOG) as f:
        lines_1 = [ln for ln in f.read().splitlines() if ln.strip()]
    batches_1 = [json.loads(ln) for ln in lines_1]
    assert len(batches_1) >= 1, "Commit log should contain at least one batch."
    all_ids_1 = []
    for expected_index, batch in enumerate(batches_1):
        assert batch["batch_index"] == expected_index, (
            "batch_index must start at 0 and increase strictly by 1; "
            f"expected {expected_index}, got {batch['batch_index']}."
        )
        assert len(batch["ids"]) <= 16, (
            f"Each batch must contain at most 16 ids (default INGEST_BATCH_SIZE); "
            f"got {len(batch['ids'])}."
        )
        all_ids_1.extend(batch["ids"])
    assert len(all_ids_1) == len(set(all_ids_1)), (
        "Commit log contains duplicate ids across batches."
    )
    assert set(all_ids_1) == expected_ids_1, (
        "Ids in commit log do not match the ids written to LanceDB after run 1."
    )

    # =================================================================== #
    # Run 2 — effectively-once across runs
    # =================================================================== #
    run2 = []
    # duplicates already stored
    for did, i in [("doc-00", 0), ("doc-05", 5), ("doc-19", 19)]:
        body = json.dumps({"id": did, "text": doc_text(i)}).encode("utf-8")
        run2.append((body, did))
    # two brand-new docs
    for i in [20, 21]:
        did = f"doc-{i:02d}"
        body = json.dumps({"id": did, "text": doc_text(i)}).encode("utf-8")
        run2.append((body, did))
    # one poison
    run2.append((b"not-json", None))

    publish_messages(run2)
    stdout2 = run_pipeline()

    w2, d2, p2 = parse_summary(stdout2)
    assert (w2, d2, p2) == (2, 3, 1), (
        "Run 2 summary mismatch. Expected written=2 skipped_duplicates=3 "
        f"dead_lettered=1, got written={w2} skipped_duplicates={d2} dead_lettered={p2}."
    )

    _, _, rows2 = load_table_rows()
    expected_ids_2 = {f"doc-{i:02d}" for i in range(22)}
    assert len(rows2) == 22, (
        f"Expected 22 rows after run 2 (no duplicates re-added), got {len(rows2)}."
    )
    assert set(rows2.keys()) == expected_ids_2, (
        f"Run 2 id set mismatch. Missing={expected_ids_2 - set(rows2.keys())}, "
        f"extra={set(rows2.keys()) - expected_ids_2}."
    )

    dlq_count2 = wait_for_count(DLQ, 4)
    assert dlq_count2 == 4, (
        f"Expected 4 total dead-lettered messages in {DLQ} after run 2, got {dlq_count2}."
    )
    main_count2 = wait_for_count(MAIN_QUEUE, 0)
    assert main_count2 == 0, (
        f"Expected main queue {MAIN_QUEUE} to be empty after run 2, got {main_count2}."
    )

    # Commit log should be appended, still monotonically increasing and covering all ids
    with open(COMMITS_LOG) as f:
        lines_2 = [ln for ln in f.read().splitlines() if ln.strip()]
    batches_2 = [json.loads(ln) for ln in lines_2]
    all_ids_2 = []
    for expected_index, batch in enumerate(batches_2):
        assert batch["batch_index"] == expected_index, (
            "batch_index across the whole commit log must be strictly increasing "
            f"by 1 starting at 0; expected {expected_index}, got {batch['batch_index']}."
        )
        all_ids_2.extend(batch["ids"])
    assert len(all_ids_2) == len(set(all_ids_2)), (
        "Commit log contains duplicate ids across batches after run 2."
    )
    assert set(all_ids_2) == expected_ids_2, (
        "Ids in commit log do not match the ids written to LanceDB after run 2."
    )
