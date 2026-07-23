import importlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"

RUN_ID = (os.environ.get("ZEALT_RUN_ID") or "local").strip() or "local"
TABLE_NAME = f"events_{RUN_ID}"
BUCKET = "lance-backup"
S3_URI = f"s3://{BUCKET}/snapshots/{TABLE_NAME}"
MISSING_S3_URI = f"s3://{BUCKET}/snapshots/does-not-exist-{RUN_ID}"

EXPECTED_COUNTS = {1: 100, 2: 130, 3: 160, 4: 160, 5: 150}
LATEST_COUNT = 150
DIM = 16


def _endpoint():
    return os.environ["AWS_ENDPOINT_URL"]


def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=_endpoint(),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def _ensure_minio():
    parsed = urlparse(_endpoint())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9000
    if not _port_open(host, port):
        helper = "/usr/local/bin/start-minio.sh"
        if os.path.isfile(helper):
            subprocess.Popen([helper], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 40
        while time.time() < deadline and not _port_open(host, port):
            time.sleep(1.0)
    assert _port_open(host, port), f"MinIO is not reachable at {host}:{port}"
    # Bucket must exist; create idempotently.
    client = _s3_client()
    try:
        client.create_bucket(Bucket=BUCKET)
    except Exception:
        pass


def _import_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    assert os.path.isfile(os.path.join(PROJECT_DIR, "solution.py")), (
        f"Expected candidate module at {PROJECT_DIR}/solution.py"
    )
    if "solution" in sys.modules:
        return importlib.reload(sys.modules["solution"])
    return importlib.import_module("solution")


@pytest.fixture(scope="session")
def restored_env():
    """Seed a versioned table, back it up to MinIO, drop the local copy, and restore."""
    _ensure_minio()
    solution = _import_solution()
    import lancedb

    workdir = tempfile.mkdtemp(prefix="lancebkp_")
    source_db = os.path.join(workdir, "source_db")
    restore_db = os.path.join(workdir, "restore_db")
    os.makedirs(source_db, exist_ok=True)
    os.makedirs(restore_db, exist_ok=True)

    # Deterministic data for ids 0..159.
    rng = np.random.default_rng(2026)
    all_vecs = rng.standard_normal((160, DIM)).astype(np.float32)
    cats = ["A", "B", "C", "D"]

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("category", pa.string()),
            pa.field("status", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
        ]
    )

    def make_rows(start, end):
        rows = []
        for i in range(start, end):
            rows.append(
                {
                    "id": i,
                    "category": cats[i % 4],
                    "status": "active",
                    "vector": all_vecs[i].tolist(),
                }
            )
        return rows

    db = lancedb.connect(source_db)
    table = db.create_table(TABLE_NAME, make_rows(0, 100), schema=schema, mode="overwrite")  # v1
    table.add(make_rows(100, 130))  # v2
    table.add(make_rows(130, 160))  # v3
    table.update(where="id < 20", values={"status": "archived"})  # v4
    table.delete("id >= 150")  # v5

    source_table_path = os.path.join(source_db, f"{TABLE_NAME}.lance")
    assert os.path.isdir(source_table_path), (
        f"source table dataset dir not created at {source_table_path}"
    )

    # Expected latest state (ids 0..149).
    expected = {}
    for i in range(0, LATEST_COUNT):
        expected[i] = {
            "category": cats[i % 4],
            "status": "archived" if i < 20 else "active",
            "vector": all_vecs[i],
        }

    # Clean any stale backup from a previous run so object counts are meaningful.
    client = _s3_client()
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"snapshots/{TABLE_NAME}"):
            for obj in page.get("Contents", []):
                client.delete_object(Bucket=BUCKET, Key=obj["Key"])
    except Exception:
        pass

    # Back up, then simulate total local data loss.
    solution.backup(source_table_path, S3_URI)
    shutil.rmtree(source_db)

    restored_table_path = os.path.join(restore_db, f"{TABLE_NAME}.lance")
    solution.restore(S3_URI, restored_table_path)

    return {
        "restore_db": restore_db,
        "restored_table_path": restored_table_path,
        "expected": expected,
        "workdir": workdir,
    }


def test_backup_uploaded_full_dataset(restored_env):
    """The backup must contain the whole dataset, including multiple version manifests."""
    client = _s3_client()
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"snapshots/{TABLE_NAME}"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    assert len(keys) > 0, f"No backup objects were uploaded under snapshots/{TABLE_NAME}"
    manifest_keys = [k for k in keys if "_versions/" in k]
    assert len(manifest_keys) >= 2, (
        "Backup must preserve the full version history: expected multiple per-version "
        f"manifest objects under _versions/, found {manifest_keys}"
    )


def test_restored_dataset_dir_exists(restored_env):
    path = restored_env["restored_table_path"]
    assert os.path.isdir(path), f"restore() did not recreate the dataset dir at {path}"


def test_latest_data_identity(restored_env):
    import lancedb

    db = lancedb.connect(restored_env["restore_db"])
    table = db.open_table(TABLE_NAME)
    assert table.count_rows() == LATEST_COUNT, (
        f"Restored latest version should have {LATEST_COUNT} rows, got {table.count_rows()}"
    )

    rows = table.to_arrow().to_pylist()
    by_id = {r["id"]: r for r in rows}
    expected = restored_env["expected"]

    assert set(by_id.keys()) == set(expected.keys()), (
        "Restored latest ids differ from the original latest ids"
    )

    for i, exp in expected.items():
        row = by_id[i]
        assert row["category"] == exp["category"], (
            f"category mismatch for id {i}: {row['category']} != {exp['category']}"
        )
        assert row["status"] == exp["status"], (
            f"status mismatch for id {i}: {row['status']} != {exp['status']} "
            "(the v4 update must survive the backup/restore)"
        )
        got_vec = np.asarray(row["vector"], dtype=np.float32)
        assert got_vec.shape == (DIM,), f"vector for id {i} has wrong shape {got_vec.shape}"
        assert np.allclose(got_vec, exp["vector"], atol=1e-5), (
            f"vector mismatch for id {i} after restore"
        )


def test_version_history_reachable(restored_env):
    import lancedb

    db = lancedb.connect(restored_env["restore_db"])
    table = db.open_table(TABLE_NAME)

    versions = table.list_versions()
    version_numbers = sorted(v["version"] for v in versions)
    for v in EXPECTED_COUNTS:
        assert v in version_numbers, (
            f"Version {v} is missing from the restored table; found versions {version_numbers}"
        )

    for v, exp_count in EXPECTED_COUNTS.items():
        table.checkout(v)
        got = table.count_rows()
        assert got == exp_count, (
            f"Restored version {v} should have {exp_count} rows, got {got}"
        )

    table.checkout_latest()
    assert table.count_rows() == LATEST_COUNT, (
        f"checkout_latest() should return {LATEST_COUNT} rows, got {table.count_rows()}"
    )


def test_negative_restore_missing_prefix(restored_env):
    import lancedb

    solution = _import_solution()
    target_dir = os.path.join(restored_env["workdir"], "missing_restore")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"missing_{RUN_ID}.lance")

    with pytest.raises(Exception):
        solution.restore(MISSING_S3_URI, target_path)

    # No usable table must be left behind.
    db = lancedb.connect(target_dir)
    with pytest.raises(Exception):
        db.open_table(f"missing_{RUN_ID}")
