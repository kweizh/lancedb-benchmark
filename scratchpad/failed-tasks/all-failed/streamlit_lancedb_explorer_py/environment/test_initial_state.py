import importlib
import importlib.util
import os

import pytest


PROJECT_DIR = "/app"
DB_DIR = "/app/db"


def test_streamlit_importable():
    try:
        importlib.import_module("streamlit")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"streamlit python package is not importable: {exc!r}")


def test_streamlit_app_test_api_importable():
    # The verifier uses streamlit.testing.v1.AppTest; make sure the symbol exists.
    spec = importlib.util.find_spec("streamlit.testing.v1")
    assert spec is not None, (
        "streamlit.testing.v1 module is required for the headless app-testing API; "
        "it is missing from the environment."
    )
    mod = importlib.import_module("streamlit.testing.v1")
    assert hasattr(mod, "AppTest"), (
        "streamlit.testing.v1.AppTest is not available; the verifier depends on it."
    )


def test_lancedb_importable():
    try:
        importlib.import_module("lancedb")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"lancedb python package is not importable: {exc!r}")


def test_pyarrow_importable():
    try:
        importlib.import_module("pyarrow")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"pyarrow python package is not importable: {exc!r}")


def test_pandas_importable():
    try:
        importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"pandas python package is not importable: {exc!r}")


def test_openai_importable():
    try:
        importlib.import_module("openai")
    except Exception as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"openai python package is not importable: {exc!r}")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), (
        f"Project directory {PROJECT_DIR} does not exist; the candidate is supposed to "
        f"create app.py inside it."
    )


def test_db_dir_exists():
    assert os.path.isdir(DB_DIR), (
        f"LanceDB database directory {DB_DIR} does not exist; the initial-state seed "
        f"pipeline should have created it."
    )


def test_db_dir_is_not_empty():
    entries = os.listdir(DB_DIR)
    assert entries, (
        f"LanceDB database directory {DB_DIR} is empty; expected pre-seeded tables "
        f"'articles' and 'cooking'."
    )


def test_seeded_tables_present_and_have_expected_shape():
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(DB_DIR)
    names = set(db.table_names())
    assert names == {"articles", "cooking"}, (
        f"Expected exactly the seeded tables {{'articles', 'cooking'}} at {DB_DIR}; "
        f"found {sorted(names)!r}."
    )

    articles = db.open_table("articles")
    assert articles.count_rows() == 50, (
        f"articles table must have 50 seeded rows; got {articles.count_rows()}."
    )

    cooking = db.open_table("cooking")
    assert cooking.count_rows() == 5, (
        f"cooking table must have 5 seeded rows; got {cooking.count_rows()}."
    )

    for tbl_name in ("articles", "cooking"):
        tbl = db.open_table(tbl_name)
        schema = tbl.schema
        names_in_schema = {f.name for f in schema}
        for required in ("id", "title", "content", "vector"):
            assert required in names_in_schema, (
                f"{tbl_name} table is missing required column {required!r}; "
                f"got columns={sorted(names_in_schema)!r}."
            )
        vec_field = schema.field("vector")
        assert pa.types.is_fixed_size_list(vec_field.type), (
            f"{tbl_name}.vector must be a fixed_size_list; got {vec_field.type!r}."
        )
        assert vec_field.type.list_size == 1536, (
            f"{tbl_name}.vector must have list size 1536; got {vec_field.type.list_size}."
        )
        assert pa.types.is_float32(vec_field.type.value_type), (
            f"{tbl_name}.vector values must be float32; got {vec_field.type.value_type!r}."
        )


def test_app_py_not_present_yet():
    # The candidate solution is responsible for creating the Streamlit app.
    app_path = os.path.join(PROJECT_DIR, "app.py")
    assert not os.path.exists(app_path), (
        f"{app_path} must not exist before the task runs; the candidate solution "
        f"must create it."
    )
