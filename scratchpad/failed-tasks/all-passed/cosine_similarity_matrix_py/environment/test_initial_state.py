import os

PROJECT_DIR = "/home/user/myproject"


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} must exist before the task starts."
    )


def test_lancedb_uri_env_var_set():
    uri = os.environ.get("LANCEDB_URI")
    assert uri, "LANCEDB_URI environment variable must be set by the entrypoint."
    assert os.path.isdir(uri), (
        f"LANCEDB_URI directory {uri} must exist (seeded by the entrypoint)."
    )


def test_lancedb_table_env_var_set():
    table = os.environ.get("LANCEDB_TABLE")
    assert table, "LANCEDB_TABLE environment variable must be set by the entrypoint."


def test_seeded_table_has_200_rows():
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    table_name = os.environ["LANCEDB_TABLE"]
    db = lancedb.connect(uri)
    assert table_name in db.table_names(), (
        f"Seeded table {table_name} must exist in LanceDB at {uri}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 200, (
        f"Seeded table must have exactly 200 rows, got {tbl.count_rows()}."
    )


def test_seeded_table_schema():
    import lancedb

    uri = os.environ["LANCEDB_URI"]
    table_name = os.environ["LANCEDB_TABLE"]
    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)
    field_names = {f.name for f in tbl.schema}
    for required in ("id", "label", "vector"):
        assert required in field_names, (
            f"Seeded table is missing required column '{required}'."
        )
