import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
LANCE_DIR = os.path.join(PROJECT_DIR, "lancedb")
CHECKPOINT = os.path.join(PROJECT_DIR, "checkpoint.json")
TABLE_NAME = "documents_index"
DIM = 32

MYSQL = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "cdc"),
    "password": os.environ.get("MYSQL_PASSWORD", "cdcpass"),
}

CATEGORIES = ["news", "tech", "sports"]


def embed(title, body):
    text = f"{title} {body}"
    v = np.zeros(DIM, dtype=np.float64)
    for tok in text.lower().split():
        bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % DIM
        v[bucket] += 1.0
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.astype(np.float32)


def connect_mysql(retries=60, delay=1.0):
    import pymysql

    last = None
    for _ in range(retries):
        try:
            return pymysql.connect(autocommit=True, **MYSQL)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay)
    raise AssertionError(f"Could not connect to local MySQL: {last}")


def exec_sql(conn, sql, args=None):
    with conn.cursor() as cur:
        cur.execute(sql, args)


def fetch_source_rows(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, body, category, price FROM appdb.documents ORDER BY id")
        rows = cur.fetchall()
    out = {}
    for rid, title, body, category, price in rows:
        out[int(rid)] = {
            "title": title,
            "body": body,
            "category": category,
            "price": float(price),
        }
    return out


def read_lance_rows():
    import lancedb

    db = lancedb.connect(LANCE_DIR)
    assert TABLE_NAME in db.table_names(), f"LanceDB table {TABLE_NAME} was not created."
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    out = {}
    for _, r in df.iterrows():
        out[int(r["id"])] = {
            "title": r["title"],
            "body": r["body"],
            "category": r["category"],
            "price": float(r["price"]),
            "vector": np.asarray(r["vector"], dtype=np.float32),
        }
    return out, tbl


def run_sync():
    res = subprocess.run(
        ["python3", "run_sync.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
    )
    return res


def parse_stats(res):
    assert res.returncode == 0, (
        f"`python3 run_sync.py` exited with {res.returncode}.\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    stats = None
    for line in reversed(res.stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            stats = obj
            break
    assert stats is not None, f"run_sync.py did not print a JSON object. STDOUT:\n{res.stdout}"
    for k in ("inserts", "updates", "deletes", "log_file", "log_pos"):
        assert k in stats, f"run_sync.py JSON missing key '{k}'. Got keys: {list(stats.keys())}"
    return stats


def read_checkpoint():
    assert os.path.isfile(CHECKPOINT), f"Checkpoint file {CHECKPOINT} does not exist."
    with open(CHECKPOINT) as f:
        cp = json.load(f)
    return cp


def checkpoint_coord(cp):
    # Accept common key spellings for the binlog coordinate.
    log_file = cp.get("log_file") or cp.get("logfile") or cp.get("file")
    log_pos = cp.get("log_pos")
    if log_pos is None:
        log_pos = cp.get("logpos", cp.get("pos"))
    assert log_file is not None and log_pos is not None, (
        f"Checkpoint must record a binlog file and position. Got: {cp}"
    )
    return str(log_file), int(log_pos)


def assert_converged(conn):
    source = fetch_source_rows(conn)
    lance, tbl = read_lance_rows()
    assert set(lance.keys()) == set(source.keys()), (
        f"LanceDB ids {sorted(lance.keys())} != MySQL ids {sorted(source.keys())}"
    )
    for rid, srow in source.items():
        lrow = lance[rid]
        assert lrow["title"] == srow["title"], f"id={rid} title mismatch: {lrow['title']!r} != {srow['title']!r}"
        assert lrow["body"] == srow["body"], f"id={rid} body mismatch"
        assert lrow["category"] == srow["category"], f"id={rid} category mismatch"
        assert abs(lrow["price"] - srow["price"]) < 1e-6, f"id={rid} price mismatch: {lrow['price']} != {srow['price']}"
        exp_vec = embed(srow["title"], srow["body"])
        got_vec = lrow["vector"]
        assert got_vec.shape == (DIM,), f"id={rid} vector length {got_vec.shape} != ({DIM},)"
        assert np.allclose(got_vec, exp_vec, atol=1e-5), f"id={rid} vector does not match deterministic embedding."
    return tbl


@pytest.fixture(scope="session", autouse=True)
def reset_environment():
    """Reset to a deterministic clean state before the verification rounds.

    Recreates an empty source table, wipes any candidate-produced LanceDB/checkpoint
    state, and rotates+purges binlogs so the earliest available binlog contains only
    the events produced by this verifier (independent of any exploration the agent
    performed while developing the solution).
    """
    conn = connect_mysql()
    exec_sql(conn, "DROP TABLE IF EXISTS appdb.documents")
    exec_sql(
        conn,
        "CREATE TABLE appdb.documents ("
        "id BIGINT PRIMARY KEY, title VARCHAR(255), body TEXT, "
        "category VARCHAR(64), price DOUBLE)",
    )
    exec_sql(conn, "FLUSH BINARY LOGS")
    with conn.cursor() as cur:
        cur.execute("SHOW BINARY LOGS")
        logs = cur.fetchall()
    newest = logs[-1][0]
    try:
        exec_sql(conn, f"PURGE BINARY LOGS TO '{newest}'")
    except Exception as e:  # noqa: BLE001
        # Non-fatal: even without purge the candidate resumes correctly; the
        # earliest binlog simply also contains prior events.
        print(f"WARN: PURGE BINARY LOGS failed: {e}")
    shutil.rmtree(LANCE_DIR, ignore_errors=True)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
    conn.close()
    yield


def _insert(conn, rid, title, body, category, price):
    exec_sql(
        conn,
        "INSERT INTO appdb.documents (id, title, body, category, price) VALUES (%s,%s,%s,%s,%s)",
        (rid, title, body, category, price),
    )


def test_round1_initial_replication():
    conn = connect_mysql()
    try:
        for rid in range(1, 11):
            _insert(
                conn,
                rid,
                f"Document title number {rid}",
                f"body text for record {rid} lorem ipsum keyword{rid}",
                CATEGORIES[rid % len(CATEGORIES)],
                round(10.5 * rid, 4),
            )
        exec_sql(
            conn,
            "UPDATE appdb.documents SET body=%s, price=%s WHERE id=%s",
            ("updated body for record 3 alpha beta gamma", 999.25, 3),
        )
        exec_sql(conn, "DELETE FROM appdb.documents WHERE id=%s", (7,))
    finally:
        conn.close()

    stats = parse_stats(run_sync())
    assert int(stats["inserts"]) == 10, f"Expected 10 inserts, got {stats['inserts']}"
    assert int(stats["updates"]) == 1, f"Expected 1 update, got {stats['updates']}"
    assert int(stats["deletes"]) == 1, f"Expected 1 delete, got {stats['deletes']}"
    read_checkpoint()


def test_round1_convergence_and_schema():
    import pyarrow as pa

    conn = connect_mysql()
    try:
        source = fetch_source_rows(conn)
        assert set(source.keys()) == {1, 2, 3, 4, 5, 6, 8, 9, 10}, (
            f"Unexpected MySQL state before convergence check: {sorted(source.keys())}"
        )
        tbl = assert_converged(conn)
    finally:
        conn.close()

    schema = tbl.schema
    fields = {f.name: f.type for f in schema}
    for name in ("id", "title", "body", "category", "price", "vector"):
        assert name in fields, f"LanceDB table missing column '{name}'."
    assert pa.types.is_integer(fields["id"]), f"id column should be integer, got {fields['id']}"
    assert pa.types.is_string(fields["title"]), f"title should be string, got {fields['title']}"
    assert pa.types.is_string(fields["body"]), f"body should be string, got {fields['body']}"
    assert pa.types.is_string(fields["category"]), f"category should be string, got {fields['category']}"
    assert pa.types.is_floating(fields["price"]), f"price should be float, got {fields['price']}"
    vec_type = fields["vector"]
    assert pa.types.is_fixed_size_list(vec_type), f"vector must be a fixed_size_list, got {vec_type}"
    assert vec_type.list_size == DIM, f"vector must have size {DIM}, got {vec_type.list_size}"


def test_round2_incremental_resume():
    before_file, before_pos = checkpoint_coord(read_checkpoint())

    conn = connect_mysql()
    try:
        for rid in (11, 12, 13):
            _insert(
                conn,
                rid,
                f"New document {rid}",
                f"fresh body content {rid} delta epsilon keyword{rid}",
                CATEGORIES[rid % len(CATEGORIES)],
                round(3.14 * rid, 4),
            )
        exec_sql(
            conn,
            "UPDATE appdb.documents SET body=%s, price=%s WHERE id=%s",
            ("second revision body for record 5", 42.5, 5),
        )
        exec_sql(
            conn,
            "UPDATE appdb.documents SET body=%s WHERE id=%s",
            ("re-updated body for record 3 zeta eta", 3),
        )
        exec_sql(conn, "DELETE FROM appdb.documents WHERE id=%s", (2,))
    finally:
        conn.close()

    stats = parse_stats(run_sync())
    assert int(stats["inserts"]) == 3, f"Expected exactly 3 inserts on resume, got {stats['inserts']}"
    assert int(stats["updates"]) == 2, f"Expected exactly 2 updates on resume, got {stats['updates']}"
    assert int(stats["deletes"]) == 1, f"Expected exactly 1 delete on resume, got {stats['deletes']}"

    after_file, after_pos = checkpoint_coord(read_checkpoint())
    advanced = (after_file != before_file) or (after_pos > before_pos)
    assert advanced, (
        f"Checkpoint did not advance. before=({before_file},{before_pos}) after=({after_file},{after_pos})"
    )


def test_round2_convergence():
    conn = connect_mysql()
    try:
        source = fetch_source_rows(conn)
        assert set(source.keys()) == {1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13}, (
            f"Unexpected MySQL state before round-2 convergence: {sorted(source.keys())}"
        )
        assert_converged(conn)
    finally:
        conn.close()


def test_round3_idempotent_noop():
    before = read_checkpoint()
    before_file, before_pos = checkpoint_coord(before)

    stats = parse_stats(run_sync())
    assert int(stats["inserts"]) == 0, f"Expected 0 inserts on no-op run, got {stats['inserts']}"
    assert int(stats["updates"]) == 0, f"Expected 0 updates on no-op run, got {stats['updates']}"
    assert int(stats["deletes"]) == 0, f"Expected 0 deletes on no-op run, got {stats['deletes']}"

    after_file, after_pos = checkpoint_coord(read_checkpoint())
    assert (after_file, after_pos) == (before_file, before_pos), (
        f"Checkpoint changed on a no-op run: ({before_file},{before_pos}) -> ({after_file},{after_pos})"
    )

    conn = connect_mysql()
    try:
        assert_converged(conn)
    finally:
        conn.close()
