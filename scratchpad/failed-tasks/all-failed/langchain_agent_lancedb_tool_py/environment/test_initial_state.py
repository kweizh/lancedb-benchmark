import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "db")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_langchain_importable():
    import langchain  # noqa: F401
    import langchain_openai  # noqa: F401
    import langchain.agents  # noqa: F401
    import langchain_core.tools  # noqa: F401


def test_openai_importable():
    import openai  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project dir {PROJECT_DIR} does not exist."


def test_db_dir_exists():
    assert os.path.isdir(DB_DIR), f"LanceDB directory {DB_DIR} does not exist."


def test_openai_api_key_present():
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY env var must be set."


def test_internal_kb_table_seeded():
    import lancedb

    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert "internal_kb" in names, (
        f"Expected 'internal_kb' table in {DB_DIR}; found {names!r}"
    )
    table = db.open_table("internal_kb")
    n = table.count_rows()
    assert n >= 30, (
        f"Expected at least 30 seeded docs in 'internal_kb', got {n}."
    )
    schema_names = {field.name for field in table.schema}
    for required in ("id", "product", "text", "vector"):
        assert required in schema_names, (
            f"Expected column '{required}' in internal_kb schema, got {schema_names!r}"
        )


def test_seeded_products_present():
    import lancedb

    db = lancedb.connect(DB_DIR)
    table = db.open_table("internal_kb")
    df = table.to_pandas()
    products = set(df["product"].unique().tolist())
    for p in ("Helios", "Aurora", "Borealis"):
        assert p in products, (
            f"Expected product '{p}' in seeded internal_kb, got {products!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
