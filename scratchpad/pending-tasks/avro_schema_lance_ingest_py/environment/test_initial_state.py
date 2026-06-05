import os
import shutil

PROJECT_DIR = "/home/user/avro_project"
AVRO_PATH = "/app/data/records.avro"


def test_python3_available():
    assert shutil.which("python3") is not None, "python3 binary not found in PATH."


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_fastavro_importable():
    import fastavro  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pandas_importable():
    import pandas  # noqa: F401


def test_project_directory_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist."
    )


def test_avro_fixture_exists():
    assert os.path.isfile(AVRO_PATH), (
        f"Pre-baked Avro fixture {AVRO_PATH} does not exist."
    )


def test_avro_fixture_has_300_records():
    import fastavro

    with open(AVRO_PATH, "rb") as f:
        reader = fastavro.reader(f)
        count = sum(1 for _ in reader)
    assert count == 300, (
        f"Avro fixture expected to contain 300 records, found {count}."
    )


def test_avro_fixture_schema_matches_expected():
    import fastavro

    with open(AVRO_PATH, "rb") as f:
        reader = fastavro.reader(f)
        schema = reader.writer_schema
    name = schema.get("name", "")
    # fastavro fully-qualifies the record name with the namespace.
    assert name.endswith("Document"), (
        f"Avro writer schema name expected to end with 'Document', got {name!r}."
    )
    field_names = [f["name"] for f in schema.get("fields", [])]
    expected = ["id", "title", "vector", "metadata"]
    assert field_names == expected, (
        f"Avro fixture top-level field order {field_names!r} does not match "
        f"expected {expected!r}."
    )


def test_lance_db_directory_clean():
    db_dir = os.path.join(PROJECT_DIR, "lance_db")
    # The LanceDB directory should NOT pre-exist; the executor creates it.
    if os.path.exists(db_dir):
        assert not os.listdir(db_dir), (
            f"LanceDB directory {db_dir} should be empty before the task starts."
        )
