import os
import importlib

import pytest

PROJECT_DIR = "/home/user/myproject"


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb could not be imported."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow could not be imported."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy could not be imported."


def test_grpc_importable():
    mod = importlib.import_module("grpc")
    assert mod is not None, "grpc (grpcio) could not be imported."


def test_grpc_tools_protoc_available():
    # grpcio-tools is required to compile the .proto into Python stubs.
    mod = importlib.import_module("grpc_tools.protoc")
    assert mod is not None, "grpc_tools.protoc (grpcio-tools) could not be imported."
