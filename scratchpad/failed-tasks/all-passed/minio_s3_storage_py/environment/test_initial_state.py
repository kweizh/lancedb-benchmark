import os
import time
import urllib.request

import pytest


WORKSPACE = "/workspace"
OUTPUT_DIR = "/workspace/output"
MINIO_HEALTH_URL = "http://127.0.0.1:9000/minio/health/ready"
MINIO_READY_TIMEOUT_SECONDS = 30


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Project directory {WORKSPACE} does not exist."


def test_output_directory_exists():
    assert os.path.isdir(OUTPUT_DIR), (
        f"Output directory {OUTPUT_DIR} must already exist before the task runs."
    )


def test_lancedb_importable():
    import lancedb  # noqa: F401

    assert hasattr(lancedb, "connect"), "lancedb.connect is not available."


def test_pyarrow_importable():
    import pyarrow as pa  # noqa: F401

    assert hasattr(pa, "schema"), "pyarrow.schema is not available."


def test_numpy_importable():
    import numpy as np  # noqa: F401

    assert hasattr(np.random, "default_rng"), "numpy.random.default_rng is not available."


def test_minio_credentials_env_vars_set():
    assert os.environ.get("MINIO_ACCESS_KEY"), (
        "MINIO_ACCESS_KEY environment variable must be set for the task."
    )
    assert os.environ.get("MINIO_SECRET_KEY"), (
        "MINIO_SECRET_KEY environment variable must be set for the task."
    )


def test_minio_server_reachable_and_healthy():
    deadline = time.time() + MINIO_READY_TIMEOUT_SECONDS
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
        f"within {MINIO_READY_TIMEOUT_SECONDS}s: {last_error}"
    )


def test_lance_bucket_precreated():
    """The Dockerfile pre-creates a bucket named 'lance-bucket' via the mc client."""
    access_key = os.environ.get("MINIO_ACCESS_KEY", "")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "")
    assert access_key and secret_key, "MinIO credentials must be set for this check."

    # Use boto3 if available, otherwise fall back to mc binary via subprocess.
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url="http://127.0.0.1:9000",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        assert "lance-bucket" in buckets, (
            f"Expected bucket 'lance-bucket' to exist on MinIO; found: {buckets}"
        )
    except ImportError:
        import shutil
        import subprocess

        mc = shutil.which("mc")
        assert mc is not None, "Neither boto3 nor mc binary is available to verify bucket."
        subprocess.run(
            [
                mc,
                "alias",
                "set",
                "localminio",
                "http://127.0.0.1:9000",
                access_key,
                secret_key,
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [mc, "ls", "localminio/"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "lance-bucket" in result.stdout, (
            f"Expected bucket 'lance-bucket' to exist on MinIO; mc ls output: {result.stdout}"
        )
