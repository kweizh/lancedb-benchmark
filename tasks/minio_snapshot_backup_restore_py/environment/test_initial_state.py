import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    assert importlib.util.find_spec("lancedb") is not None, "lancedb is not installed."


def test_boto3_importable():
    assert importlib.util.find_spec("boto3") is not None, "boto3 is not installed."


def test_numpy_importable():
    assert importlib.util.find_spec("numpy") is not None, "numpy is not installed."


def test_pyarrow_importable():
    assert importlib.util.find_spec("pyarrow") is not None, "pyarrow is not installed."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_minio_env_vars_present():
    for var in (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "ALLOW_HTTP",
    ):
        assert os.environ.get(var), f"Expected environment variable {var} to be set."


def test_minio_running_and_bucket_exists():
    boto3 = pytest.importorskip("boto3")
    endpoint = os.environ["AWS_ENDPOINT_URL"]
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    buckets = client.list_buckets()
    names = {b["Name"] for b in buckets.get("Buckets", [])}
    assert "lance-backup" in names, (
        f"Expected MinIO bucket 'lance-backup' to exist, found: {sorted(names)}"
    )
