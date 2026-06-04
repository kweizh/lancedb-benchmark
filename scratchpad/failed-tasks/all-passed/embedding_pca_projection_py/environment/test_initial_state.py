import os

import pytest


PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_sklearn_importable():
    import sklearn.decomposition  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project dir {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), f"LanceDB dir {LANCEDB_DIR} does not exist."


def test_source_articles_table_present_with_600_rows():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    assert "articles" in names, f"Source table 'articles' missing; tables = {names}."
    tbl = db.open_table("articles")
    n = tbl.count_rows()
    assert n == 600, f"Source 'articles' table must have exactly 600 rows; got {n}."


def test_source_articles_schema_is_128d():
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("articles")
    schema = tbl.schema
    fields = {f.name: f.type for f in schema}
    assert "id" in fields, "Source table missing 'id' column."
    assert "title" in fields, "Source table missing 'title' column."
    assert "embedding" in fields, "Source table missing 'embedding' column."

    emb_type = fields["embedding"]
    assert pa.types.is_fixed_size_list(emb_type), (
        f"Source 'embedding' must be a fixed-size list; got {emb_type}."
    )
    assert emb_type.list_size == 128, (
        f"Source 'embedding' must be 128-dimensional; got list_size={emb_type.list_size}."
    )


def test_zealt_run_id_env_var_present():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable must be set in the task environment."


def test_pca_model_artifact_not_pre_created():
    # The candidate is supposed to create this file. The initial environment must NOT ship it.
    assert not os.path.exists("/app/pca_model.npz"), (
        "/app/pca_model.npz must not exist before the candidate runs."
    )
