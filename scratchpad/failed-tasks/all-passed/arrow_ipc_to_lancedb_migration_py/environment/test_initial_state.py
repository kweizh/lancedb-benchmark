import importlib
import os
import pytest

PROJECT_DIR = "/home/user/myproject"
SOURCE_PATH = "/app/source/dataset.arrows"
QUERY_VECTOR_PATH = "/app/query_vector.npy"


def test_lancedb_importable():
    mod = importlib.import_module("lancedb")
    assert mod is not None, "lancedb is not installed or not importable."


def test_pyarrow_importable():
    mod = importlib.import_module("pyarrow")
    assert mod is not None, "pyarrow is not installed or not importable."


def test_numpy_importable():
    mod = importlib.import_module("numpy")
    assert mod is not None, "numpy is not installed or not importable."


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_source_arrow_ipc_file_exists():
    assert os.path.isfile(SOURCE_PATH), (
        f"Pre-built Arrow IPC stream file {SOURCE_PATH} is missing."
    )


def test_query_vector_file_exists():
    assert os.path.isfile(QUERY_VECTOR_PATH), (
        f"Precomputed query vector {QUERY_VECTOR_PATH} is missing."
    )


def test_source_arrow_stream_has_expected_row_count_and_schema():
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.OSFile(SOURCE_PATH, "rb") as sink:
        with ipc.open_stream(sink) as reader:
            table = reader.read_all()

    assert table.num_rows == 2000, (
        f"Source IPC stream must contain 2000 rows; got {table.num_rows}."
    )

    field_names = [f.name for f in table.schema]
    for expected in ("id", "text", "tags", "embedding", "created_at"):
        assert expected in field_names, (
            f"Source schema is missing required column '{expected}'."
        )

    emb_type = table.schema.field("embedding").type
    assert pa.types.is_fixed_size_list(emb_type), (
        "Source 'embedding' column must be a fixed_size_list (got "
        f"{emb_type})."
    )
    assert emb_type.list_size == 48, (
        f"Source 'embedding' fixed_size_list width must be 48; got {emb_type.list_size}."
    )
    assert pa.types.is_float32(emb_type.value_type), (
        "Source 'embedding' value type must be float32 (got "
        f"{emb_type.value_type})."
    )


def test_query_vector_has_correct_shape_and_dtype():
    import numpy as np

    arr = np.load(QUERY_VECTOR_PATH)
    assert arr.shape == (48,), f"Query vector must have shape (48,); got {arr.shape}."
    assert arr.dtype == np.float32, (
        f"Query vector must be float32; got {arr.dtype}."
    )
