import importlib
import os
import shutil
import socket

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"lancedb could not be imported: {exc}")


def test_confluent_kafka_importable():
    try:
        importlib.import_module("confluent_kafka")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"confluent_kafka could not be imported: {exc}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"pyarrow could not be imported: {exc}")


def test_rpk_binary_available():
    assert shutil.which("rpk") is not None, "rpk (Redpanda CLI) binary not found in PATH."


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_redpanda_broker_reachable():
    # The Redpanda broker is started by the container entrypoint and should be
    # accepting Kafka connections on localhost:9092.
    last_err = None
    for _ in range(30):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect(("127.0.0.1", 9092))
            sock.close()
            return
        except OSError as exc:  # pragma: no cover - retry loop
            last_err = exc
            try:
                sock.close()
            except OSError:
                pass
            import time

            time.sleep(1)
    pytest.fail(f"Redpanda broker not reachable on 127.0.0.1:9092: {last_err}")
