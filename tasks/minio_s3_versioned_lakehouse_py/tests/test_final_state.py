import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
FIXTURES_DIR = "/app/fixtures"
VERSIONS_PATH = os.path.join(PROJECT_DIR, "versions.json")

MINIO_ENDPOINT = "http://127.0.0.1:9000"
BUCKET = "lancedb-lakehouse"
DB_URI = "s3://lancedb-lakehouse/db"
TABLE = "documents"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
REGION = "us-east-1"
K = 5

STORAGE_OPTIONS = {
    "endpoint": MINIO_ENDPOINT,
    "region": REGION,
    "allow_http": "true",
    "aws_access_key_id": ACCESS_KEY,
    "aws_secret_access_key": SECRET_KEY,
}


# --------------------------------------------------------------------------
# Fixture data / ground-truth helpers
# --------------------------------------------------------------------------
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _base_rows():
    return _load_json(os.path.join(FIXTURES_DIR, "base.json"))


def _added_rows():
    return _load_json(os.path.join(FIXTURES_DIR, "added.json"))


def _queries():
    return _load_json(os.path.join(FIXTURES_DIR, "queries.json"))


def _snapshot(label):
    base = _base_rows()
    added = _added_rows()
    if label == "base":
        return list(base)
    combined = list(base) + list(added)
    if label == "added":
        return combined
    if label == "deleted":
        return [r for r in combined if r.get("category") != "legacy"]
    raise ValueError(f"unknown snapshot {label}")


def _brute_force_topk(rows, qvec, k):
    q = np.asarray(qvec, dtype=np.float64)
    scored = []
    for r in rows:
        v = np.asarray(r["vector"], dtype=np.float64)
        dist = float(np.sum((v - q) ** 2))
        scored.append((dist, int(r["id"])))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [rid for _, rid in scored[:k]]


# --------------------------------------------------------------------------
# Build the lakehouse once for the whole test module
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built():
    if os.path.exists(VERSIONS_PATH):
        os.remove(VERSIONS_PATH)
    result = subprocess.run(
        [sys.executable, "run.py", "build"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("BUILD STDOUT:\n", result.stdout)
    print("BUILD STDERR:\n", result.stderr)
    assert result.returncode == 0, (
        f"'python3 run.py build' must exit with status 0, got {result.returncode}. "
        f"stderr: {result.stderr}"
    )
    assert os.path.isfile(VERSIONS_PATH), (
        f"build did not create {VERSIONS_PATH}"
    )
    return _load_json(VERSIONS_PATH)


def _run_query(query_name, version, k):
    result = subprocess.run(
        [sys.executable, "run.py", "query",
         "--query", query_name, "--version", str(version), "--k", str(k)],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"query command failed (rc={result.returncode}): {result.stderr}"
    )
    parsed = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "ids" in obj and "version" in obj:
            parsed = obj
    assert parsed is not None, (
        f"query did not print a JSON object with keys 'version' and 'ids'. "
        f"stdout: {result.stdout!r}"
    )
    return parsed


# --------------------------------------------------------------------------
# 1. Data physically stored on MinIO (storage_options handling)
# --------------------------------------------------------------------------
def test_data_stored_on_minio(built):
    subprocess.run(
        ["mc", "alias", "set", "verihost", MINIO_ENDPOINT, ACCESS_KEY, SECRET_KEY],
        capture_output=True, text=True, timeout=60,
    )
    result = subprocess.run(
        ["mc", "ls", "--recursive", f"verihost/{BUCKET}"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"'mc ls' failed: {result.stderr}"
    listing = result.stdout
    print("MINIO LISTING:\n", listing)
    assert "documents.lance" in listing, (
        "Expected the LanceDB table to be stored under 'db/documents.lance/' on MinIO. "
        "This proves storage_options (custom endpoint/credentials) were used instead of local disk."
    )
    assert ".lance" in listing, (
        "Expected at least one '*.lance' data file in the MinIO bucket."
    )


# --------------------------------------------------------------------------
# 2. versions.json shape
# --------------------------------------------------------------------------
def test_versions_json_shape(built):
    vs = built
    for key in ("base", "added", "deleted", "latest"):
        assert key in vs, f"versions.json missing key '{key}'"
        assert isinstance(vs[key], int), f"versions.json['{key}'] must be an integer"
    assert vs["base"] >= 1, "base version must be >= 1"
    assert vs["added"] > vs["base"], "added version must be greater than base version"
    assert vs["deleted"] > vs["added"], "deleted version must be greater than added version"
    assert vs["latest"] > vs["deleted"], (
        "latest version must be greater than deleted version, proving optimize() created a new version"
    )


# --------------------------------------------------------------------------
# 3. Independent version inspection via a short-lived Python process
# --------------------------------------------------------------------------
def test_independent_version_inspection(built):
    helper = r'''
import json, os, sys
import lancedb

STORAGE_OPTIONS = {
    "endpoint": "%s",
    "region": "%s",
    "allow_http": "true",
    "aws_access_key_id": "%s",
    "aws_secret_access_key": "%s",
}
db = lancedb.connect("%s", storage_options=STORAGE_OPTIONS)
t = db.open_table("%s")
versions = t.list_versions()
with open("%s") as f:
    vs = json.load(f)
out = {"num_versions": len(versions)}
for label in ["base", "added", "deleted"]:
    t.checkout(vs[label])
    out[label + "_count"] = int(t.count_rows())
t.checkout_latest()
sys.stdout.write("RESULT_JSON:" + json.dumps(out) + "\n")
sys.stdout.flush()
os._exit(0)
''' % (MINIO_ENDPOINT, REGION, ACCESS_KEY, SECRET_KEY, DB_URI, TABLE, VERSIONS_PATH)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(helper)
        helper_path = fh.name

    try:
        result = subprocess.run(
            [sys.executable, helper_path],
            capture_output=True, text=True, timeout=300,
        )
    finally:
        os.remove(helper_path)

    print("INSPECT STDOUT:\n", result.stdout)
    print("INSPECT STDERR:\n", result.stderr)

    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            payload = json.loads(line[len("RESULT_JSON:"):])
    assert payload is not None, (
        f"independent inspection produced no RESULT_JSON. stderr: {result.stderr}"
    )

    assert payload["num_versions"] >= 4, (
        f"expected at least 4 versions on the S3 table, got {payload['num_versions']}"
    )

    n_base = len(_base_rows())
    n_added = len(_added_rows())
    n_legacy = sum(
        1 for r in (_base_rows() + _added_rows()) if r.get("category") == "legacy"
    )

    assert payload["base_count"] == n_base, (
        f"base version should have {n_base} rows, got {payload['base_count']}"
    )
    assert payload["added_count"] == n_base + n_added, (
        f"added version should have {n_base + n_added} rows, got {payload['added_count']}"
    )
    assert payload["deleted_count"] == n_base + n_added - n_legacy, (
        f"deleted version should have {n_base + n_added - n_legacy} rows, "
        f"got {payload['deleted_count']}"
    )


# --------------------------------------------------------------------------
# 4. Time-travel query correctness: q_added
# --------------------------------------------------------------------------
def test_time_travel_q_added(built):
    vs = built
    qvec = _queries()["q_added"]

    expected_base = _brute_force_topk(_snapshot("base"), qvec, K)
    expected_added = _brute_force_topk(_snapshot("added"), qvec, K)

    # Sanity: the fixtures must make time travel observable.
    assert expected_base != expected_added, (
        "fixture sanity check failed: q_added top-k identical across base/added snapshots"
    )

    res_base = _run_query("q_added", vs["base"], K)
    res_added = _run_query("q_added", vs["added"], K)

    assert res_base["version"] == vs["base"], "returned version must echo the requested version"
    assert res_base["ids"] == expected_base, (
        f"time-travel to base version returned {res_base['ids']}, expected {expected_base}"
    )
    assert res_added["ids"] == expected_added, (
        f"time-travel to added version returned {res_added['ids']}, expected {expected_added}"
    )


# --------------------------------------------------------------------------
# 5. Time-travel query correctness: q_deleted
# --------------------------------------------------------------------------
def test_time_travel_q_deleted(built):
    vs = built
    qvec = _queries()["q_deleted"]

    expected_added = _brute_force_topk(_snapshot("added"), qvec, K)
    expected_deleted = _brute_force_topk(_snapshot("deleted"), qvec, K)

    assert expected_added != expected_deleted, (
        "fixture sanity check failed: q_deleted top-k identical across added/deleted snapshots"
    )

    res_added = _run_query("q_deleted", vs["added"], K)
    res_deleted = _run_query("q_deleted", vs["deleted"], K)

    assert res_added["ids"] == expected_added, (
        f"time-travel to added version returned {res_added['ids']}, expected {expected_added}"
    )
    assert res_deleted["ids"] == expected_deleted, (
        f"time-travel to deleted version returned {res_deleted['ids']}, expected {expected_deleted}"
    )

    # The nearest neighbor to q_deleted is a 'legacy' row: present before delete, gone after.
    legacy_ids = {
        int(r["id"]) for r in (_base_rows() + _added_rows()) if r.get("category") == "legacy"
    }
    assert res_added["ids"][0] in legacy_ids, (
        "expected the nearest neighbor to q_deleted (before delete) to be a legacy row"
    )
    assert res_deleted["ids"][0] not in legacy_ids, (
        "after deleting legacy rows, the top hit for q_deleted must not be a legacy row"
    )


# --------------------------------------------------------------------------
# 6. Ordering / tie-break contract
# --------------------------------------------------------------------------
def test_result_ordering_contract(built):
    vs = built
    qvec = _queries()["q_added"]
    res = _run_query("q_added", vs["latest"], K)

    rows = {int(r["id"]): r for r in _snapshot("deleted")}
    # Only ids that still exist at latest (post-delete) should appear.
    dists = []
    for rid in res["ids"]:
        assert rid in rows, f"returned id {rid} is not present in the latest snapshot"
        v = np.asarray(rows[rid]["vector"], dtype=np.float64)
        dists.append(float(np.sum((v - np.asarray(qvec, dtype=np.float64)) ** 2)))
    assert dists == sorted(dists), (
        f"returned ids must be ordered nearest-first by ascending L2 distance, got distances {dists}"
    )
