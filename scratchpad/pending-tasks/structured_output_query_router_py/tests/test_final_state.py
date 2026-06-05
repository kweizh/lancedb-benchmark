import importlib
import json
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_PATH = os.path.join(PROJECT_DIR, ".expected.json")


# --- helpers ---------------------------------------------------------------


def _load_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        return importlib.reload(sys.modules["solution"])
    return importlib.import_module("solution")


@pytest.fixture(scope="module")
def solution_module():
    mod = _load_solution()
    assert hasattr(mod, "route_and_search"), "solution.py must expose route_and_search."
    return mod


@pytest.fixture(scope="module")
def expected_anchors():
    with open(EXPECTED_PATH) as f:
        payload = json.load(f)
    return {a["query"]: a for a in payload["anchors"]}


def _check_envelope(out, k):
    assert isinstance(out, dict), f"route_and_search must return a dict, got {type(out)}"
    assert set(out.keys()) == {"mode", "filters", "results"}, (
        f"top-level keys must be exactly mode/filters/results, got {sorted(out.keys())}"
    )
    assert out["mode"] in {"vector", "fts", "hybrid", "filter_only"}, (
        f"mode {out['mode']} not in allowed set"
    )
    assert isinstance(out["filters"], dict), "filters must be a dict"
    assert isinstance(out["results"], list), "results must be a list"
    assert len(out["results"]) <= k, f"results length {len(out['results'])} exceeds k={k}"
    for r in out["results"]:
        assert isinstance(r, dict), "each result item must be a dict"
        assert "id" in r, "result rows must contain an 'id' field"
        assert isinstance(r["id"], int), f"id must be int, got {type(r['id'])}"


def _approx(value, expected, tol=0.5):
    return value is not None and abs(float(value) - float(expected)) <= tol


# --- table sanity ----------------------------------------------------------


def test_table_schema_and_rowcount():
    import lancedb

    run_id = os.environ["ZEALT_RUN_ID"]
    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table(f"products_{run_id}")
    assert tbl.count_rows() == 500, "products table should still contain 500 rows."
    names = {f.name for f in tbl.schema}
    assert {"id", "sku", "name", "description", "category", "price", "release_date", "vector"}.issubset(names)
    # vector width == 1536
    vector_field = next(f for f in tbl.schema if f.name == "vector")
    assert vector_field.type.list_size == 1536, (
        f"vector column must be fixed_size_list<float32, 1536>, got {vector_field.type}"
    )


# --- anchor queries --------------------------------------------------------


def test_anchor_1_filter_only_under_50(solution_module, expected_anchors):
    q = "products under $50"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "filter_only", f"expected filter_only, got {out['mode']}"
    assert _approx(out["filters"].get("price_max"), 50.0), (
        f"filters.price_max should be ~50, got {out['filters'].get('price_max')}"
    )
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_2_fts_abc_123(solution_module, expected_anchors):
    q = "find item ABC-123"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "fts", f"expected fts, got {out['mode']}"
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_3_vector_running_shoes(solution_module, expected_anchors):
    q = "comfortable running shoes"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "vector", f"expected vector, got {out['mode']}"
    for key in ("category", "price_min", "price_max", "date_min", "date_max"):
        assert out["filters"].get(key) in (None, ""), (
            f"filters.{key} should be null for a pure vector query, got {out['filters'].get(key)}"
        )
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_4_hybrid_red_shoes_under_100(solution_module, expected_anchors):
    q = "red shoes under $100"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "hybrid", f"expected hybrid, got {out['mode']}"
    assert _approx(out["filters"].get("price_max"), 100.0), (
        f"filters.price_max should be ~100, got {out['filters'].get('price_max')}"
    )
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_5_filter_only_clothing_range(solution_module, expected_anchors):
    q = "items in clothing category between $200 and $500"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "filter_only", f"expected filter_only, got {out['mode']}"
    cat = out["filters"].get("category")
    assert cat and cat.lower() == "clothing", f"filters.category should be 'clothing', got {cat}"
    assert _approx(out["filters"].get("price_min"), 200.0), (
        f"filters.price_min should be ~200, got {out['filters'].get('price_min')}"
    )
    assert _approx(out["filters"].get("price_max"), 500.0), (
        f"filters.price_max should be ~500, got {out['filters'].get('price_max')}"
    )
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_6_fts_xyz_789(solution_module, expected_anchors):
    q = "search for code XYZ-789"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "fts", f"expected fts, got {out['mode']}"
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_7_vector_earbuds(solution_module, expected_anchors):
    q = "wireless bluetooth earbuds"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "vector", f"expected vector, got {out['mode']}"
    for key in ("category", "price_min", "price_max", "date_min", "date_max"):
        assert out["filters"].get(key) in (None, ""), (
            f"filters.{key} should be null for a pure vector query, got {out['filters'].get(key)}"
        )
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )


def test_anchor_8_hybrid_leather_wallet(solution_module, expected_anchors):
    q = "premium leather wallet under $300"
    out = solution_module.route_and_search(q, k=5)
    _check_envelope(out, k=5)
    assert out["mode"] == "hybrid", f"expected hybrid, got {out['mode']}"
    assert _approx(out["filters"].get("price_max"), 300.0), (
        f"filters.price_max should be ~300, got {out['filters'].get('price_max')}"
    )
    target = expected_anchors[q]["target_id"]
    assert out["results"][0]["id"] == target, (
        f"top-1 id {out['results'][0]['id']} != expected {target}"
    )
