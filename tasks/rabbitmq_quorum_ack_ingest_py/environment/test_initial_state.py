import os
import shutil

import pytest

PROJECT_DIR = "/home/user/project"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_pika_importable():
    import pika  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_rabbitmq_server_binary_available():
    assert (
        shutil.which("rabbitmq-server") is not None
    ), "rabbitmq-server binary not found in PATH."


def test_rabbitmqctl_binary_available():
    assert (
        shutil.which("rabbitmqctl") is not None
    ), "rabbitmqctl binary not found in PATH."


def test_project_dir_exists():
    assert os.path.isdir(
        PROJECT_DIR
    ), f"Project directory {PROJECT_DIR} does not exist."
