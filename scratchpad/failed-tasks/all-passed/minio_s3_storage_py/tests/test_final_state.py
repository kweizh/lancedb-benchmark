import json
import os
import time
import urllib.request

import numpy as np
import pyarrow as pa
import pytest


OUTPUT_FILE = "/workspace/output/s3_results.json"
MINIO_HEALTH_URL = "http://127.0.0.1:9000/minio/health/ready"
MINIO_ENDPOINT = "http://127.0.0.1:9000"
MINIO_REGION = "us-east-1"
BUCKET_URI = "s3://lance-bucket/"
TABLE_NAME = "vectors_s3"
SEED = 11
NUM_ROWS = 16
VECTOR_DIM = 8
TOP_K = 3


def _wait_for_minio(timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(MINIO_HEALTH_URL, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
                last_error = f"unexpected status {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1)
    pytest.fail(
        f"MinIO server did not become healthy at {MINIO_HEALTH_URL} "
        f"within {timeout_seconds}s: {last_error}"
    )


def _expected_top_ids():
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((NUM_ROWS, VECTOR_DIM)).astype("float32")
    query = rng.standard_normal(VECTOR_DIM).astype("float32")
    dists = np.sum((vectors - query) ** 2, axis=1)
    order = np.argsort(dists, kind="stable")
    return [int(i) for i in order[:TOP_K]]


def _storage_options():
    access = os.environ["MINIO_ACCESS_KEY"]
    secret = os.environ["MINIO_SECRET_KEY"]
    return {
        "endpoint": MINIO_ENDPOINT,
        "region": MINIO_REGION,
        "aws_access_key_id": access,
        "aws_secret_access_key": secret,
        "allow_http": "true",
    }


@pytest.fixture(scope="session", autouse=True)
def _ensure_minio_ready():
    _wait_for_minio()
    yield


def test_output_file_exists():
    assert os.path.isfile(OUTPUT_FILE), f"Expected output file at {OUTPUT_FILE}, but it does not exist."


def test_output_file_is_top3_with_required_keys():
    with open(OUTPUT_FILE) as f:
        results = json.load(f)
    assert isinstance(results, list), f"Output must be a JSON list, got {type(results).__name__}."
    assert len(results) == TOP_K, f"Expected exactly {TOP_K} results, got {len(results)}: {results}"
    for i, item in enumerate(results):
        assert isinstance(item, dict), f"Result item {i} is not an object: {item!r}"
        assert "id" in item, f"Result item {i} missing 'id' field: {item!r}"
        assert "payload" in item, f"Result item {i} missing 'payload' field: {item!r}"
        assert "_distance" in item, f"Result item {i} missing '_distance' field: {item!r}"
        assert isinstance(item["id"], int), f"Result {i} 'id' must be int, got {type(item['id']).__name__}."
        assert isinstance(item["payload"], str), (
            f"Result {i} 'payload' must be str, got {type(item['payload']).__name__}."
        )
        assert isinstance(item["_distance"], (int, float)), (
            f"Result {i} '_distance' must be numeric, got {type(item['_distance']).__name__}."
        )


def test_top_ids_match_ground_truth():
    with open(OUTPUT_FILE) as f:
        results = json.load(f)
    actual_ids = [int(r["id"]) for r in results]
    expected_ids = _expected_top_ids()
    assert actual_ids == expected_ids, (
        f"Top-{TOP_K} ids do not match deterministic ground truth. "
        f"Expected {expected_ids}, got {actual_ids}."
    )


def test_payload_matches_id_format():
    with open(OUTPUT_FILE) as f:
        results = json.load(f)
    for r in results:
        expected_payload = f"row-{int(r['id']):02d}"
        assert r["payload"] == expected_payload, (
            f"Payload mismatch for id={r['id']}: expected {expected_payload!r}, got {r['payload']!r}."
        )


def test_table_on_minio_has_16_rows():
    import lancedb

    db = lancedb.connect(BUCKET_URI, storage_options=_storage_options())
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"Expected table {TABLE_NAME!r} in MinIO-backed LanceDB; found tables: {table_names}."
    )
    tbl = db.open_table(TABLE_NAME)
    n = tbl.count_rows()
    assert n == NUM_ROWS, f"Expected {NUM_ROWS} rows in table {TABLE_NAME!r}, got {n}."


def test_table_schema_is_correct():
    import lancedb

    db = lancedb.connect(BUCKET_URI, storage_options=_storage_options())
    tbl = db.open_table(TABLE_NAME)
    schema = tbl.schema
    field_names = {f.name for f in schema}
    for required in ("id", "payload", "vector"):
        assert required in field_names, (
            f"Table {TABLE_NAME!r} missing required field {required!r}. Got fields: {field_names}."
        )
    vector_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type), (
        f"Field 'vector' must be a fixed_size_list, got {vector_field.type}."
    )
    assert vector_field.type.list_size == VECTOR_DIM, (
        f"Field 'vector' must have list_size={VECTOR_DIM}, got {vector_field.type.list_size}."
    )
    assert pa.types.is_float32(vector_field.type.value_type), (
        f"Field 'vector' element type must be float32, got {vector_field.type.value_type}."
    )
