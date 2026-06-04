import json
import math
import os
import socket

import pytest
import requests
from bs4 import BeautifulSoup
from xprocess import ProcessStarter

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_FIXTURE = os.path.join(PROJECT_DIR, ".expected.json")
HOST = "127.0.0.1"
PORT = 5000
BASE_URL = f"http://{HOST}:{PORT}"


def _load_fixture():
    with open(EXPECTED_FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def fixture():
    return _load_fixture()


@pytest.fixture(scope="session")
def new_product_id():
    # Use ZEALT_RUN_ID for parallel-run safety; fall back to a static suffix.
    run_id = os.environ.get("ZEALT_RUN_ID", "local").strip() or "local"
    return f"prod-new-{run_id}"


@pytest.fixture(scope="session")
def start_app(xprocess):
    class Starter(ProcessStarter):
        name = "flask_crud_app"
        args = ["python3", "app.py"]
        env = os.environ.copy()
        popen_kwargs = {
            "cwd": PROJECT_DIR,
            "text": True,
        }
        timeout = 120
        terminate_on_interrupt = True

        def startup_check(self):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((HOST, PORT)) == 0

    xprocess.ensure(Starter.name, Starter)
    yield
    info = xprocess.getinfo(Starter.name)
    info.terminate()


def _get_json_products():
    resp = requests.get(f"{BASE_URL}/api/products", timeout=30)
    assert resp.status_code == 200, (
        f"GET /api/products returned status {resp.status_code}: {resp.text[:500]}"
    )
    ctype = resp.headers.get("Content-Type", "")
    assert "application/json" in ctype, (
        f"Expected Content-Type to include 'application/json', got: {ctype}"
    )
    payload = resp.json()
    assert isinstance(payload, list), f"GET /api/products must return a JSON list, got {type(payload)}"
    return payload


def _by_id(items, key="id"):
    return {item[key]: item for item in items if key in item}


def _get_index_rows():
    """Return a list of dicts, one per data <tr> in the GET / HTML table."""
    resp = requests.get(f"{BASE_URL}/", timeout=30)
    assert resp.status_code == 200, f"GET / returned status {resp.status_code}: {resp.text[:500]}"
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    assert table is not None, "GET / HTML response is missing a <table> element."
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        cell_texts = [c.get_text(strip=True) for c in cells]
        # Filter out header rows (containing <th> with all-text labels) heuristically.
        if any(c.name == "th" for c in cells):
            continue
        if not cell_texts:
            continue
        rows.append({
            "text": "\n".join(cell_texts),
            "html": str(tr),
            "cells": cell_texts,
        })
    return rows


def test_initial_index_lists_seeded_products(start_app, fixture):
    seeded = fixture["seeded_products"]
    seeded_ids = {p["id"] for p in seeded}
    rows = _get_index_rows()
    # Every seeded product must appear by id in some data row's text content.
    found_ids = set()
    for p in seeded:
        for row in rows:
            if p["id"] in row["text"] and p["name"] in row["text"] and p["category"] in row["text"]:
                # price may render with or without trailing zeros; compare numerically when possible.
                # Simpler: check that the numeric price token appears as substring in some form.
                price_str = f"{p['price']}"
                if price_str in row["text"] or f"{p['price']:.2f}" in row["text"]:
                    found_ids.add(p["id"])
                    break
    assert found_ids == seeded_ids, (
        f"Index page is missing seeded product rows.\n"
        f"  expected ids: {seeded_ids}\n"
        f"  found ids:    {found_ids}\n"
        f"  rows: {[r['text'] for r in rows]}"
    )


def test_initial_api_products_matches_seed(start_app, fixture):
    seeded = fixture["seeded_products"]
    seeded_by_id = _by_id(seeded)
    items = _get_json_products()
    assert len(items) == len(seeded), (
        f"Expected {len(seeded)} products in /api/products initially, got {len(items)}: {items!r}"
    )
    items_by_id = _by_id(items)
    assert set(items_by_id.keys()) == set(seeded_by_id.keys()), (
        f"Mismatched product ids in /api/products.\n"
        f"  expected: {set(seeded_by_id.keys())}\n"
        f"  got:      {set(items_by_id.keys())}"
    )
    for pid, expected in seeded_by_id.items():
        got = items_by_id[pid]
        assert got["name"] == expected["name"], (
            f"Product {pid} name mismatch: expected {expected['name']!r}, got {got['name']!r}"
        )
        assert got["category"] == expected["category"], (
            f"Product {pid} category mismatch: expected {expected['category']!r}, got {got['category']!r}"
        )
        assert math.isclose(float(got["price"]), float(expected["price"]), rel_tol=1e-6, abs_tol=1e-6), (
            f"Product {pid} price mismatch: expected {expected['price']!r}, got {got['price']!r}"
        )


def test_edit_form_is_prefilled_for_prod_003(start_app, fixture):
    seeded_by_id = _by_id(fixture["seeded_products"])
    target = seeded_by_id["prod-003"]
    resp = requests.get(f"{BASE_URL}/product/prod-003/edit", timeout=30)
    assert resp.status_code == 200, (
        f"GET /product/prod-003/edit returned {resp.status_code}: {resp.text[:500]}"
    )
    body = resp.text
    assert target["name"] in body, (
        f"Edit form for prod-003 must be pre-filled with name {target['name']!r}; body: {body[:1000]}"
    )
    assert target["category"] in body, (
        f"Edit form for prod-003 must be pre-filled with category {target['category']!r}; body: {body[:1000]}"
    )
    # The seeded price should appear somewhere in the rendered HTML (e.g., as input value).
    price_tokens = [str(target["price"]), f"{target['price']:.2f}"]
    assert any(tok in body for tok in price_tokens), (
        f"Edit form for prod-003 must show price {target['price']!r}; body: {body[:1000]}"
    )


def test_create_product_appears_in_listing_and_api(start_app, new_product_id):
    form = {
        "id": new_product_id,
        "name": "Brand New Widget",
        "category": "gadgets",
        "price": "19.95",
    }
    resp = requests.post(f"{BASE_URL}/product", data=form, timeout=30, allow_redirects=False)
    assert 200 <= resp.status_code < 400, (
        f"POST /product (create) returned {resp.status_code}: {resp.text[:500]}"
    )
    items = _get_json_products()
    items_by_id = _by_id(items)
    assert new_product_id in items_by_id, (
        f"Newly created product {new_product_id!r} is missing from /api/products: {items!r}"
    )
    got = items_by_id[new_product_id]
    assert got["name"] == "Brand New Widget", (
        f"New product name mismatch: {got['name']!r}"
    )
    assert got["category"] == "gadgets", (
        f"New product category mismatch: {got['category']!r}"
    )
    assert math.isclose(float(got["price"]), 19.95, rel_tol=1e-6, abs_tol=1e-6), (
        f"New product price mismatch: {got['price']!r}"
    )
    # Also visible in the HTML listing.
    rows = _get_index_rows()
    assert any(new_product_id in r["text"] and "Brand New Widget" in r["text"] for r in rows), (
        f"New product {new_product_id!r} is missing from the HTML listing rows: {[r['text'] for r in rows]}"
    )


def test_update_existing_product_price(start_app, fixture):
    seeded_by_id = _by_id(fixture["seeded_products"])
    target = seeded_by_id["prod-001"]
    form = {
        "name": target["name"],
        "category": target["category"],
        "price": "999.99",
    }
    resp = requests.post(f"{BASE_URL}/product/prod-001", data=form, timeout=30, allow_redirects=False)
    assert 200 <= resp.status_code < 400, (
        f"POST /product/prod-001 (update) returned {resp.status_code}: {resp.text[:500]}"
    )
    items_by_id = _by_id(_get_json_products())
    assert "prod-001" in items_by_id, "prod-001 disappeared after update."
    assert math.isclose(float(items_by_id["prod-001"]["price"]), 999.99, rel_tol=1e-6, abs_tol=1e-3), (
        f"prod-001 price not updated to 999.99: got {items_by_id['prod-001']['price']!r}"
    )
    rows = _get_index_rows()
    assert any("prod-001" in r["text"] and ("999.99" in r["text"] or "999.9" in r["text"]) for r in rows), (
        f"Updated price for prod-001 not visible in HTML listing: {[r['text'] for r in rows]}"
    )


def test_delete_existing_product(start_app):
    resp = requests.post(f"{BASE_URL}/product/prod-002/delete", timeout=30, allow_redirects=False)
    assert 200 <= resp.status_code < 400, (
        f"POST /product/prod-002/delete returned {resp.status_code}: {resp.text[:500]}"
    )
    items_by_id = _by_id(_get_json_products())
    assert "prod-002" not in items_by_id, (
        f"prod-002 still present in /api/products after delete: {items_by_id!r}"
    )
    rows = _get_index_rows()
    assert not any("prod-002" in r["text"] for r in rows), (
        f"prod-002 still rendered in the HTML listing after delete: {[r['text'] for r in rows]}"
    )


def test_lancedb_on_disk_reflects_changes(start_app, new_product_id):
    """Reopen LanceDB directly and confirm create/update/delete persisted."""
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table("products")
    df = tbl.to_pandas()
    ids = set(df["id"].tolist())

    # Create persisted.
    assert new_product_id in ids, (
        f"Newly created product {new_product_id!r} not found in on-disk LanceDB table: {sorted(ids)}"
    )
    new_row = df[df["id"] == new_product_id].iloc[0]
    assert new_row["name"] == "Brand New Widget", (
        f"On-disk new product name mismatch: {new_row['name']!r}"
    )

    # Update persisted.
    assert "prod-001" in ids, "prod-001 missing on disk after update."
    upd_row = df[df["id"] == "prod-001"].iloc[0]
    assert math.isclose(float(upd_row["price"]), 999.99, rel_tol=1e-6, abs_tol=1e-3), (
        f"On-disk prod-001 price not updated: {upd_row['price']!r}"
    )

    # Delete persisted.
    assert "prod-002" not in ids, (
        f"prod-002 still present on disk after delete: {sorted(ids)}"
    )
