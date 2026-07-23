import hashlib
import os
import shutil
import subprocess
import time

import lancedb
import pymongo
import pytest

PROJECT_DIR = "/home/user/project"
LANCE_DIR = os.path.join(PROJECT_DIR, "lancedb")
TOKEN_FILE = os.path.join(PROJECT_DIR, "resume_token.json")
TABLE_NAME = "documents"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = "cdc"
COLL_NAME = "documents"

# Give change streams a moment to surface freshly committed events before draining.
PROPAGATION_SLEEP = 2.0


def expected_vector(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    return [b / 255.0 for b in digest]


def vec_matches(actual, text):
    exp = expected_vector(text)
    if actual is None:
        return False
    actual = list(actual)
    if len(actual) != len(exp):
        return False
    return all(abs(float(a) - e) <= 1e-4 for a, e in zip(actual, exp))


def run_sync():
    env = os.environ.copy()
    env["MONGO_URI"] = MONGO_URI
    return subprocess.run(
        ["python3", "sync.py"],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def read_rows():
    db = lancedb.connect(LANCE_DIR)
    if TABLE_NAME not in db.table_names():
        return {}
    tbl = db.open_table(TABLE_NAME)
    rows = tbl.to_arrow().to_pylist()
    return {r["id"]: r for r in rows}


@pytest.fixture(scope="session")
def scenario():
    # Ensure the local MongoDB single-node replica set is up (idempotent helper).
    startup = subprocess.run(
        ["start-mongo.sh"], capture_output=True, text=True, timeout=180
    )
    assert startup.returncode == 0, (
        f"start-mongo.sh failed to bring up MongoDB: {startup.stdout}\n{startup.stderr}"
    )

    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)
    # Ensure a writable primary is available.
    client.admin.command("hello")

    results = {}

    # ---- Clean state ----
    client.drop_database(DB_NAME)
    if os.path.isdir(LANCE_DIR):
        shutil.rmtree(LANCE_DIR)
    if os.path.isfile(TOKEN_FILE):
        os.remove(TOKEN_FILE)

    coll = client[DB_NAME][COLL_NAME]

    # ---- Phase 0: establish resume position on an empty run ----
    proc0 = run_sync()
    results["rc0"] = proc0.returncode
    results["stderr0"] = proc0.stderr
    results["token_exists0"] = os.path.isfile(TOKEN_FILE)

    # ---- Phase 1: initial inserts ----
    coll.insert_one({"_id": "d1", "text": "hello world", "category": "a"})
    coll.insert_one({"_id": "d2", "text": "vector database", "category": "b"})
    coll.insert_one({"_id": "d3", "text": "change stream", "category": "a"})
    time.sleep(PROPAGATION_SLEEP)
    proc1 = run_sync()
    results["rc1"] = proc1.returncode
    results["stderr1"] = proc1.stderr
    results["snap1"] = read_rows()

    # ---- Phase 2: update d2, delete d3, insert d4 ----
    coll.update_one({"_id": "d2"}, {"$set": {"text": "updated text"}})
    coll.delete_one({"_id": "d3"})
    coll.insert_one({"_id": "d4", "text": "fresh doc", "category": "c"})
    time.sleep(PROPAGATION_SLEEP)
    proc2 = run_sync()
    results["rc2"] = proc2.returncode
    results["stderr2"] = proc2.stderr
    results["snap2"] = read_rows()

    # ---- Phase 3: full replacement of d1 ----
    coll.replace_one({"_id": "d1"}, {"text": "replaced content", "category": "z"})
    time.sleep(PROPAGATION_SLEEP)
    proc3 = run_sync()
    results["rc3"] = proc3.returncode
    results["stderr3"] = proc3.stderr
    results["snap3"] = read_rows()

    # ---- Phase 4: insert then delete same id within one batch ----
    coll.insert_one({"_id": "d5", "text": "temp", "category": "a"})
    coll.delete_one({"_id": "d5"})
    time.sleep(PROPAGATION_SLEEP)
    proc4 = run_sync()
    results["rc4"] = proc4.returncode
    results["stderr4"] = proc4.stderr
    results["snap4"] = read_rows()

    # ---- Phase 5: re-run with no new events (idempotency / crash recovery) ----
    proc5 = run_sync()
    results["rc5"] = proc5.returncode
    results["stderr5"] = proc5.stderr
    results["snap5"] = read_rows()
    results["token_exists5"] = os.path.isfile(TOKEN_FILE)

    client.close()
    return results


def test_empty_run_establishes_resume_token(scenario):
    assert scenario["rc0"] == 0, (
        f"Initial `python3 sync.py` run failed (rc={scenario['rc0']}). "
        f"stderr: {scenario['stderr0']}"
    )
    assert scenario["token_exists0"], (
        f"Resume token file {TOKEN_FILE} was not created on the initial empty run; "
        "a starting resume position must be persisted even with zero events."
    )


def test_inserts_are_synced(scenario):
    assert scenario["rc1"] == 0, (
        f"sync.py failed after inserts (rc={scenario['rc1']}). stderr: {scenario['stderr1']}"
    )
    rows = scenario["snap1"]
    assert set(rows.keys()) == {"d1", "d2", "d3"}, (
        f"Expected exactly rows d1, d2, d3 after inserts, got: {sorted(rows.keys())}"
    )
    expected = {
        "d1": ("hello world", "a"),
        "d2": ("vector database", "b"),
        "d3": ("change stream", "a"),
    }
    for rid, (text, category) in expected.items():
        assert rows[rid]["text"] == text, (
            f"Row {rid} text mismatch: expected {text!r}, got {rows[rid]['text']!r}"
        )
        assert rows[rid]["category"] == category, (
            f"Row {rid} category mismatch: expected {category!r}, got {rows[rid]['category']!r}"
        )
        assert vec_matches(rows[rid]["vector"], text), (
            f"Row {rid} vector does not match embedding of {text!r}."
        )


def test_update_delete_insert_applied(scenario):
    assert scenario["rc2"] == 0, (
        f"sync.py failed after update/delete/insert (rc={scenario['rc2']}). "
        f"stderr: {scenario['stderr2']}"
    )
    rows = scenario["snap2"]
    assert set(rows.keys()) == {"d1", "d2", "d4"}, (
        f"Expected exactly rows d1, d2, d4, got: {sorted(rows.keys())}"
    )
    assert "d3" not in rows, "Row d3 should have been deleted (tombstone) but still exists."

    assert rows["d2"]["text"] == "updated text", (
        f"d2 text should be 'updated text', got {rows['d2']['text']!r}"
    )
    assert vec_matches(rows["d2"]["vector"], "updated text"), (
        "d2 vector was not updated to match the new text 'updated text'."
    )

    assert rows["d4"]["text"] == "fresh doc", (
        f"d4 text should be 'fresh doc', got {rows['d4']['text']!r}"
    )
    assert rows["d4"]["category"] == "c", (
        f"d4 category should be 'c', got {rows['d4']['category']!r}"
    )
    assert vec_matches(rows["d4"]["vector"], "fresh doc"), (
        "d4 vector does not match embedding of 'fresh doc'."
    )

    # d1 is unchanged at this phase.
    assert rows["d1"]["text"] == "hello world", (
        f"d1 should still be 'hello world' at this phase, got {rows['d1']['text']!r}"
    )
    assert vec_matches(rows["d1"]["vector"], "hello world"), (
        "d1 vector should still match embedding of 'hello world'."
    )


def test_replace_operation_applied(scenario):
    assert scenario["rc3"] == 0, (
        f"sync.py failed after replace (rc={scenario['rc3']}). stderr: {scenario['stderr3']}"
    )
    rows = scenario["snap3"]
    assert "d1" in rows, "Row d1 missing after replace operation."
    assert rows["d1"]["text"] == "replaced content", (
        f"d1 text should be 'replaced content' after replace, got {rows['d1']['text']!r}"
    )
    assert rows["d1"]["category"] == "z", (
        f"d1 category should be 'z' after replace, got {rows['d1']['category']!r}"
    )
    assert vec_matches(rows["d1"]["vector"], "replaced content"), (
        "d1 vector was not updated to match the replaced text 'replaced content'."
    )


def test_same_run_insert_then_delete_yields_no_row(scenario):
    assert scenario["rc4"] == 0, (
        f"sync.py failed on ordered batch (rc={scenario['rc4']}). stderr: {scenario['stderr4']}"
    )
    rows = scenario["snap4"]
    assert "d5" not in rows, (
        "Row d5 was inserted then deleted within one run and must not exist, but it does."
    )


def test_rerun_is_idempotent(scenario):
    assert scenario["rc5"] == 0, (
        f"Re-running sync.py with no new events failed (rc={scenario['rc5']}). "
        f"stderr: {scenario['stderr5']}"
    )
    assert scenario["token_exists5"], f"Resume token file {TOKEN_FILE} missing after re-run."

    before = scenario["snap4"]
    after = scenario["snap5"]
    assert set(after.keys()) == {"d1", "d2", "d4"}, (
        f"After idempotent re-run expected exactly d1, d2, d4, got: {sorted(after.keys())}"
    )
    assert set(before.keys()) == set(after.keys()), (
        "Re-running with no new events changed which rows exist "
        f"(before: {sorted(before.keys())}, after: {sorted(after.keys())})."
    )
    for rid in after:
        assert before[rid]["text"] == after[rid]["text"], (
            f"Row {rid} text changed after an idempotent re-run "
            f"({before[rid]['text']!r} -> {after[rid]['text']!r}); events were reapplied."
        )
        assert vec_matches(after[rid]["vector"], after[rid]["text"]), (
            f"Row {rid} vector inconsistent with its text after re-run."
        )
