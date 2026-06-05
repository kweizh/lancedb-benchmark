import importlib
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")


def test_run_id_env_var_set():
    assert os.environ.get("ZEALT_RUN_ID"), "ZEALT_RUN_ID environment variable must be set."


def test_lancedb_importable():
    importlib.import_module("lancedb")


def test_pyarrow_importable():
    importlib.import_module("pyarrow")


def test_numpy_importable():
    importlib.import_module("numpy")


def test_pandas_importable():
    importlib.import_module("pandas")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(DB_DIR), f"LanceDB data directory {DB_DIR} does not exist."


def test_seeded_products_table_present_and_correct():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"products_{run_id}"

    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert table_name in names, f"Seeded table {table_name} not found. Found: {names}"

    tbl = db.open_table(table_name)
    df = tbl.to_pandas()
    assert len(df) == 1000, f"Expected 1000 seeded rows, got {len(df)}"

    expected_cols = {"id", "brand", "category", "color", "in_stock", "price", "vector"}
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing required columns. Have: {set(df.columns)}"
    )

    # Cardinalities
    assert df["brand"].nunique() == 10, f"Expected 10 brands, got {df['brand'].nunique()}"
    assert df["category"].nunique() == 6, f"Expected 6 categories, got {df['category'].nunique()}"
    assert df["color"].nunique() == 8, f"Expected 8 colors, got {df['color'].nunique()}"
    assert set(df["in_stock"].unique()).issubset({True, False}), "in_stock must be boolean."

    # Vector dim
    sample = df["vector"].iloc[0]
    assert len(sample) == 32, f"Expected 32-d vectors, got {len(sample)}"
