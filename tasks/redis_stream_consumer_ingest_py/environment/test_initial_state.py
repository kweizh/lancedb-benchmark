import importlib
import shutil
import time

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_project_dir_exists():
    import os

    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb could not be imported."


def test_redis_client_importable():
    mod = importlib.import_module("redis")
    assert mod is not None, "redis (redis-py) could not be imported."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy could not be imported."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow could not be imported."


def test_redis_server_binary_available():
    assert shutil.which("redis-server") is not None, "redis-server binary not found in PATH."


def test_local_redis_reachable():
    redis = importlib.import_module("redis")
    client = redis.Redis(host="127.0.0.1", port=6379, socket_connect_timeout=2)
    last_err = None
    for _ in range(15):
        try:
            if client.ping():
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1)
    pytest.fail(f"Local redis-server not reachable at 127.0.0.1:6379: {last_err}")
