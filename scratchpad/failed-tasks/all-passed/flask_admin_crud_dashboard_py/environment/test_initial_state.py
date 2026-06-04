import importlib
import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_FIXTURE = os.path.join(PROJECT_DIR, ".expected.json")


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_data_dir_exists():
    assert os.path.isdir(LANCEDB_DIR), (
        f"Seeded LanceDB directory {LANCEDB_DIR} does not exist; the build-time seed step did not run."
    )


def test_expected_fixture_exists_and_is_well_formed():
    assert os.path.isfile(EXPECTED_FIXTURE), (
        f"Build-time fixture {EXPECTED_FIXTURE} is missing; "
        "the verifier needs it to know the expected seeded products."
    )
    with open(EXPECTED_FIXTURE) as f:
        payload = json.load(f)
    assert isinstance(payload, dict), "Fixture must be a JSON object."
    seeded = payload.get("seeded_products")
    assert isinstance(seeded, list) and len(seeded) == 5, (
        "Fixture must contain a 'seeded_products' list of length 5."
    )
    seen_ids = set()
    for entry in seeded:
        assert isinstance(entry, dict), "Each seeded product entry must be a JSON object."
        for key in ("id", "name", "category", "price"):
            assert key in entry, f"Seeded product entry missing key '{key}': {entry!r}"
        assert isinstance(entry["id"], str) and entry["id"], "Product id must be a non-empty string."
        assert isinstance(entry["name"], str) and entry["name"], "Product name must be a non-empty string."
        assert isinstance(entry["category"], str) and entry["category"], "Product category must be a non-empty string."
        assert isinstance(entry["price"], (int, float)), "Product price must be numeric."
        seen_ids.add(entry["id"])
    assert len(seen_ids) == 5, f"Seeded product ids must be unique; got {seen_ids!r}"


def test_lancedb_importable():
    importlib.import_module("lancedb")


def test_flask_importable():
    importlib.import_module("flask")


def test_lancedb_products_table_has_5_rows():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    assert "products" in db.table_names(), "LanceDB table 'products' was not seeded."
    tbl = db.open_table("products")
    n = tbl.count_rows()
    assert n == 5, f"Expected exactly 5 seeded rows in 'products', got {n}."


def test_verifier_deps_importable():
    importlib.import_module("xprocess")
    importlib.import_module("requests")
    importlib.import_module("bs4")
