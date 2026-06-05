import importlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")


# ---------- Helpers ----------

BRANDS = [f"Brand{c}" for c in "ABCDEFGHIJ"]
CATEGORIES = [f"Cat{i+1}" for i in range(6)]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]


def _query_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(32).astype(np.float32)


def _table_name() -> str:
    run_id = os.environ["ZEALT_RUN_ID"]
    return f"products_{run_id}"


def _load_full_df() -> pd.DataFrame:
    import lancedb

    db = lancedb.connect(DB_DIR)
    return db.open_table(_table_name()).to_pandas()


def _build_sql_predicate(facets: dict) -> str:
    """Mirror of the reference predicate used to cross-check candidate's server-side filter."""
    clauses = []
    if "brand" in facets and facets["brand"]:
        vals = ", ".join("'" + v.replace("'", "''") + "'" for v in facets["brand"])
        clauses.append(f"brand IN ({vals})")
    if "category" in facets and facets["category"]:
        vals = ", ".join("'" + v.replace("'", "''") + "'" for v in facets["category"])
        clauses.append(f"category IN ({vals})")
    if "color" in facets and facets["color"]:
        vals = ", ".join("'" + v.replace("'", "''") + "'" for v in facets["color"])
        clauses.append(f"color IN ({vals})")
    if "in_stock" in facets:
        clauses.append(f"in_stock = {str(bool(facets['in_stock'])).lower()}")
    if "price_max" in facets:
        clauses.append(f"price <= {float(facets['price_max'])}")
    if "price_min" in facets:
        clauses.append(f"price >= {float(facets['price_min'])}")
    return " AND ".join(clauses)


def _ground_truth(df: pd.DataFrame, qv: np.ndarray, facets: dict, k: int):
    mask = pd.Series([True] * len(df))
    if "brand" in facets and facets["brand"]:
        mask &= df["brand"].isin(facets["brand"])
    if "category" in facets and facets["category"]:
        mask &= df["category"].isin(facets["category"])
    if "color" in facets and facets["color"]:
        mask &= df["color"].isin(facets["color"])
    if "in_stock" in facets:
        mask &= df["in_stock"] == bool(facets["in_stock"])
    if "price_max" in facets:
        mask &= df["price"] <= float(facets["price_max"])
    if "price_min" in facets:
        mask &= df["price"] >= float(facets["price_min"])

    sub = df[mask].copy()
    if len(sub) == 0:
        top_ids = []
    else:
        vecs = np.stack(sub["vector"].to_list()).astype(np.float32)
        dists = np.sum((vecs - qv[None, :]) ** 2, axis=1)
        sub = sub.assign(_dist=dists)
        sub = sub.sort_values(by=["_dist", "id"], ascending=[True, True])
        top_ids = sub["id"].head(k).astype(int).tolist()

    # Facet counts over the full filtered set
    counts = {
        "brand": sub["brand"].value_counts().to_dict() if len(sub) else {},
        "category": sub["category"].value_counts().to_dict() if len(sub) else {},
        "color": sub["color"].value_counts().to_dict() if len(sub) else {},
        "in_stock": {
            "true": int((sub["in_stock"] == True).sum()) if len(sub) else 0,
            "false": int((sub["in_stock"] == False).sum()) if len(sub) else 0,
        },
    }
    return top_ids, counts, int(len(sub))


@pytest.fixture(scope="session")
def solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"solution.py not found at {SOLUTION_PATH}"
    # Clear any stale cache and ensure fresh import
    pycache = os.path.join(PROJECT_DIR, "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache, ignore_errors=True)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


@pytest.fixture(scope="session")
def full_df():
    return _load_full_df()


# ---------- Tests ----------

def test_solution_has_faceted_search(solution_module):
    assert hasattr(solution_module, "faceted_search"), (
        "solution.py must expose a top-level function `faceted_search`."
    )


def test_no_op_facets_returns_top_k_and_full_counts(solution_module, full_df):
    qv = _query_vec(seed=1)
    out = solution_module.faceted_search(qv.tolist(), {}, 10)

    assert set(out.keys()) >= {"results", "facet_counts"}, (
        f"Return value missing required keys; got {list(out.keys())}"
    )

    assert len(out["results"]) == 10, f"Expected 10 results, got {len(out['results'])}"
    for r in out["results"]:
        for key in ("id", "brand", "category", "color", "in_stock", "price", "distance"):
            assert key in r, f"Result missing key '{key}': {r}"
        assert "vector" not in r, "Results must not include the raw `vector` field."

    gt_ids, gt_counts, gt_total = _ground_truth(full_df, qv, {}, 10)
    got_ids = [int(r["id"]) for r in out["results"]]
    assert got_ids == gt_ids, f"Top-10 id order mismatch.\n  got={got_ids}\n  expected={gt_ids}"

    fc = out["facet_counts"]
    assert sum(fc["brand"].values()) == 1000, (
        f"brand counts must sum to 1000 on empty filter; got {sum(fc['brand'].values())}"
    )
    assert sum(fc["category"].values()) == 1000
    assert sum(fc["color"].values()) == 1000
    is_counts = fc["in_stock"]
    assert int(is_counts.get("true", 0)) + int(is_counts.get("false", 0)) == 1000


def test_single_field_brand_include_list(solution_module, full_df):
    qv = _query_vec(seed=2)
    facets = {"brand": ["BrandA", "BrandB"]}
    out = solution_module.faceted_search(qv.tolist(), facets, 5)

    for r in out["results"]:
        assert r["brand"] in {"BrandA", "BrandB"}, f"Filter leaked: {r}"

    gt_ids, gt_counts, gt_total = _ground_truth(full_df, qv, facets, 5)
    got_ids = [int(r["id"]) for r in out["results"]]
    assert got_ids == gt_ids, f"Top-5 id order mismatch.\n  got={got_ids}\n  expected={gt_ids}"

    brand_counts = out["facet_counts"]["brand"]
    assert set(brand_counts.keys()).issubset({"BrandA", "BrandB"}), (
        f"facet_counts.brand keys must be subset of filtered values; got {brand_counts}"
    )
    assert sum(int(v) for v in brand_counts.values()) == gt_total, (
        f"brand facet counts must sum to filtered-set size {gt_total}; got {brand_counts}"
    )


def test_multi_field_and_filter(solution_module, full_df):
    qv = _query_vec(seed=3)
    facets = {
        "brand": ["BrandA", "BrandC", "BrandE"],
        "color": ["red", "blue"],
        "in_stock": True,
        "price_max": 250.0,
    }
    out = solution_module.faceted_search(qv.tolist(), facets, 10)

    for r in out["results"]:
        assert r["brand"] in {"BrandA", "BrandC", "BrandE"}, f"brand leaked: {r}"
        assert r["color"] in {"red", "blue"}, f"color leaked: {r}"
        assert r["in_stock"] is True or r["in_stock"] == 1, f"in_stock leaked: {r}"
        assert r["price"] <= 250.0 + 1e-9, f"price leaked: {r}"

    gt_ids, gt_counts, gt_total = _ground_truth(full_df, qv, facets, 10)
    got_ids = [int(r["id"]) for r in out["results"]]
    assert got_ids == gt_ids, f"Top-10 id order mismatch.\n  got={got_ids}\n  expected={gt_ids}"

    fc = out["facet_counts"]
    # Brand counts
    for b, c in gt_counts["brand"].items():
        assert int(fc["brand"].get(b, 0)) == int(c), (
            f"brand count mismatch for {b}: got {fc['brand'].get(b)} expected {c}"
        )
    # Color counts
    for col, c in gt_counts["color"].items():
        assert int(fc["color"].get(col, 0)) == int(c), (
            f"color count mismatch for {col}: got {fc['color'].get(col)} expected {c}"
        )


def test_tight_filter_smaller_than_k(solution_module, full_df):
    qv = _query_vec(seed=4)
    facets = {
        "brand": ["BrandJ"],
        "category": ["Cat6"],
        "color": ["orange"],
        "in_stock": False,
        "price_max": 30.0,
    }
    out = solution_module.faceted_search(qv.tolist(), facets, 20)

    gt_ids, gt_counts, gt_total = _ground_truth(full_df, qv, facets, 20)
    got_ids = [int(r["id"]) for r in out["results"]]
    assert len(got_ids) == len(gt_ids), (
        f"Expected {len(gt_ids)} rows for tight filter, got {len(got_ids)}"
    )
    assert set(got_ids) == set(gt_ids), (
        f"Id-set mismatch.\n  got={set(got_ids)}\n  expected={set(gt_ids)}"
    )

    fc = out["facet_counts"]
    if gt_total > 0:
        assert int(fc["brand"].get("BrandJ", 0)) == gt_total
        assert int(fc["category"].get("Cat6", 0)) == gt_total
        assert int(fc["color"].get("orange", 0)) == gt_total


def test_determinism(solution_module):
    qv = _query_vec(seed=5)
    facets = {"brand": ["BrandD"], "in_stock": True}
    a = solution_module.faceted_search(qv.tolist(), facets, 7)
    b = solution_module.faceted_search(qv.tolist(), facets, 7)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    ), "Two identical calls produced different outputs."


def test_server_side_filter_matches_lancedb_where(solution_module, full_df):
    """Verify the candidate result id-set matches what LanceDB's own .where() returns."""
    import lancedb

    qv = _query_vec(seed=6)
    facets = {
        "category": ["Cat2", "Cat4"],
        "color": ["green"],
        "price_min": 50.0,
        "price_max": 400.0,
    }

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(_table_name())
    sql = _build_sql_predicate(facets)

    k = 8
    server_df = tbl.search(qv).where(sql).limit(k).to_pandas()
    server_ids = set(int(x) for x in server_df["id"].tolist())

    out = solution_module.faceted_search(qv.tolist(), facets, k)
    got_ids = set(int(r["id"]) for r in out["results"])

    assert got_ids == server_ids, (
        f"Candidate's id-set must match LanceDB server-side filter for facets {facets}.\n"
        f"  got={got_ids}\n  expected={server_ids}"
    )
