import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
PRODUCTS_PATH = os.path.join(PROJECT_DIR, "products.json")


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_voyageai_importable():
    import voyageai  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_products_json_exists():
    assert os.path.isfile(PRODUCTS_PATH), (
        f"Catalogue file {PRODUCTS_PATH} should be pre-seeded by the environment."
    )


def test_products_json_has_60_entries_with_required_fields():
    with open(PRODUCTS_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list), "products.json must be a JSON array."
    assert len(data) == 60, f"Expected 60 products, found {len(data)}."
    for i, row in enumerate(data):
        assert isinstance(row, dict), f"Row {i} is not an object."
        assert "id" in row and isinstance(row["id"], str), f"Row {i} missing string `id`."
        assert "description" in row and isinstance(row["description"], str), (
            f"Row {i} missing string `description`."
        )


def test_anchor_product_ids_present():
    with open(PRODUCTS_PATH) as f:
        data = json.load(f)
    ids = {row["id"] for row in data}
    required = {
        "prod-electronics-headphones-anc",
        "prod-kitchen-espresso-machine",
        "prod-fitness-carbon-marathon-shoes",
        "prod-books-dragon-wizard-fantasy",
    }
    missing = required - ids
    assert not missing, f"Anchor product ids missing from products.json: {missing}"


def test_voyage_api_key_env_var_present():
    assert os.environ.get("VOYAGE_API_KEY"), (
        "VOYAGE_API_KEY must be exported in the task container."
    )


def test_zealt_run_id_env_var_present():
    assert os.environ.get("ZEALT_RUN_ID"), (
        "ZEALT_RUN_ID must be exported in the task container."
    )


def test_lancedb_data_dir_exists_empty_or_absent():
    # The lancedb data directory may exist (created by the env) but must not
    # already contain the candidate's products table.
    data_dir = os.path.join(PROJECT_DIR, "lancedb_data")
    if os.path.isdir(data_dir):
        run_id = os.environ.get("ZEALT_RUN_ID", "")
        if run_id:
            table_dir = os.path.join(data_dir, f"products_{run_id}.lance")
            assert not os.path.isdir(table_dir), (
                f"Candidate output table {table_dir} must not pre-exist."
            )


def test_candidate_outputs_absent():
    for fname in ("solution.py", "run_search.py"):
        path = os.path.join(PROJECT_DIR, fname)
        assert not os.path.exists(path), (
            f"{path} should be created by the candidate, not pre-seeded."
        )
