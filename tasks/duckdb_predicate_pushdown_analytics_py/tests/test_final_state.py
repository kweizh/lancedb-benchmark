import csv
import json
import os
import subprocess

import numpy as np
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.jsonl")
CATEGORIES_PATH = os.path.join(DATA_DIR, "categories.csv")
LANCEDB_PATH = os.path.join(PROJECT_DIR, "lancedb")
RUN_PY = os.path.join(PROJECT_DIR, "run.py")

DIST_TOL = 1e-3
PRICE_TOL = 1e-6
AVG_TOL = 1e-4


# --------------------------------------------------------------------------
# Helpers to independently reproduce the expected pipeline from the raw files.
# --------------------------------------------------------------------------
def _load_documents():
    docs = []
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        for ln in f.read().splitlines():
            if ln.strip():
                docs.append(json.loads(ln))
    return docs


def _load_categories():
    mapping = {}
    with open(CATEGORIES_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["category"]] = row["department"]
    return mapping


def _sq_l2(query, vec):
    q = np.asarray(query, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    diff = q - v
    return float(np.dot(diff, diff))


def _expected(query, top_k, max_price, category=None):
    docs = _load_documents()
    cat_to_dept = _load_categories()

    scored = []
    for d in docs:
        scored.append((_sq_l2(query, d["vector"]), int(d["id"]), d))
    # top-K nearest candidate pool, ties broken by ascending id
    scored.sort(key=lambda t: (t[0], t[1]))
    pool = scored[:top_k]

    survivors = []
    for dist, doc_id, d in pool:
        if not bool(d["in_stock"]):
            continue
        if float(d["price"]) > max_price:
            continue
        if category is not None and d["category"] != category:
            continue
        survivors.append(
            {
                "id": doc_id,
                "title": d["title"],
                "category": d["category"],
                "department": cat_to_dept[d["category"]],
                "price": float(d["price"]),
                "distance": dist,
            }
        )

    survivors.sort(key=lambda h: (h["distance"], h["id"]))

    # per-department row_number window (ordered by distance then id)
    dept_counter = {}
    hits = []
    for h in survivors:
        dept = h["department"]
        dept_counter[dept] = dept_counter.get(dept, 0) + 1
        hit = dict(h)
        hit["dept_rank"] = dept_counter[dept]
        hits.append(hit)

    # per-department aggregation
    departments = []
    for dept in sorted(dept_counter.keys()):
        members = [h for h in survivors if h["department"] == dept]
        avg_price = round(sum(m["price"] for m in members) / len(members), 4)
        departments.append(
            {
                "department": dept,
                "num_docs": len(members),
                "avg_price": avg_price,
                "min_distance": min(m["distance"] for m in members),
            }
        )

    return {"hits": hits, "departments": departments}


def _run_cli(args):
    result = subprocess.run(
        ["python3", RUN_PY] + args,
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    assert result.returncode == 0, (
        f"run.py exited with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    out = result.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        start = out.find("{")
        end = out.rfind("}")
        assert start != -1 and end != -1 and end > start, (
            f"stdout is not valid JSON and no JSON object could be extracted.\nSTDOUT:\n{result.stdout}"
        )
        return json.loads(out[start : end + 1])


def _assert_hits_match(actual, expected):
    assert isinstance(actual, dict), f"Top-level output must be a JSON object, got {type(actual)}."
    assert set(actual.keys()) == {"hits", "departments"}, (
        f"Output must have exactly keys 'hits' and 'departments'; got {sorted(actual.keys())}."
    )
    a_hits = actual["hits"]
    e_hits = expected["hits"]
    assert isinstance(a_hits, list), "'hits' must be a JSON array."
    assert len(a_hits) == len(e_hits), (
        f"Expected {len(e_hits)} hits, got {len(a_hits)}. "
        f"Expected ids={[h['id'] for h in e_hits]}, got ids={[h.get('id') for h in a_hits]}."
    )
    for i, (ah, eh) in enumerate(zip(a_hits, e_hits)):
        assert set(ah.keys()) == {
            "id",
            "title",
            "category",
            "department",
            "price",
            "distance",
            "dept_rank",
        }, f"hit[{i}] has wrong keys: {sorted(ah.keys())}."
        assert ah["id"] == eh["id"], f"hit[{i}] id mismatch: expected {eh['id']}, got {ah['id']}."
        assert ah["title"] == eh["title"], f"hit[{i}] title mismatch for id {eh['id']}."
        assert ah["category"] == eh["category"], f"hit[{i}] category mismatch for id {eh['id']}."
        assert ah["department"] == eh["department"], (
            f"hit[{i}] department mismatch for id {eh['id']}: expected {eh['department']}, got {ah['department']}."
        )
        assert abs(float(ah["price"]) - eh["price"]) <= PRICE_TOL, (
            f"hit[{i}] price mismatch for id {eh['id']}: expected {eh['price']}, got {ah['price']}."
        )
        assert abs(float(ah["distance"]) - eh["distance"]) <= DIST_TOL, (
            f"hit[{i}] distance mismatch for id {eh['id']}: expected ~{eh['distance']}, got {ah['distance']}."
        )
        assert ah["dept_rank"] == eh["dept_rank"], (
            f"hit[{i}] dept_rank mismatch for id {eh['id']}: expected {eh['dept_rank']}, got {ah['dept_rank']}."
        )


def _assert_departments_match(actual, expected):
    a_deps = actual["departments"]
    e_deps = expected["departments"]
    assert isinstance(a_deps, list), "'departments' must be a JSON array."
    assert len(a_deps) == len(e_deps), (
        f"Expected {len(e_deps)} departments, got {len(a_deps)}. "
        f"Expected={[d['department'] for d in e_deps]}, got={[d.get('department') for d in a_deps]}."
    )
    for i, (ad, ed) in enumerate(zip(a_deps, e_deps)):
        assert set(ad.keys()) == {"department", "num_docs", "avg_price", "min_distance"}, (
            f"department[{i}] has wrong keys: {sorted(ad.keys())}."
        )
        assert ad["department"] == ed["department"], (
            f"department[{i}] name mismatch: expected {ed['department']}, got {ad['department']}."
        )
        assert ad["num_docs"] == ed["num_docs"], (
            f"department[{i}] ({ed['department']}) num_docs mismatch: expected {ed['num_docs']}, got {ad['num_docs']}."
        )
        assert abs(float(ad["avg_price"]) - ed["avg_price"]) <= AVG_TOL, (
            f"department[{i}] ({ed['department']}) avg_price mismatch: expected {ed['avg_price']}, got {ad['avg_price']}."
        )
        assert abs(float(ad["min_distance"]) - ed["min_distance"]) <= DIST_TOL, (
            f"department[{i}] ({ed['department']}) min_distance mismatch: expected ~{ed['min_distance']}, got {ad['min_distance']}."
        )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_run_py_exists():
    assert os.path.isfile(RUN_PY), f"Expected solution CLI at {RUN_PY}."


def test_lancedb_table_populated_with_float32_vectors():
    import lancedb

    # Ensure the solution has materialized its LanceDB table by exercising the
    # (rerunnable) CLI once; a lazily-built table would otherwise not exist yet.
    _run_cli(
        ["--query-vector", "0.9,0.1,0.0,0.0,0.5,0.2,0.1,0.3", "--top-k", "20", "--max-price", "80.0"]
    )

    docs = _load_documents()
    db = lancedb.connect(LANCEDB_PATH)
    names = list(db.table_names())
    assert names, f"No LanceDB tables were found at {LANCEDB_PATH}."

    matched = None
    for name in names:
        tbl = db.open_table(name)
        schema = tbl.schema
        try:
            n_rows = tbl.count_rows()
        except Exception:
            n_rows = len(tbl.to_pandas())
        has_f32_vec = False
        for field in schema:
            if pa.types.is_fixed_size_list(field.type) and pa.types.is_float32(field.type.value_type):
                has_f32_vec = True
                break
        if has_f32_vec and n_rows == len(docs):
            matched = (name, n_rows)
            break

    assert matched is not None, (
        f"Expected a LanceDB table with {len(docs)} rows and a fixed_size_list<float32> vector column; "
        f"tables present: {names}."
    )


def test_basic_hybrid_query_no_category_filter():
    query = [0.9, 0.1, 0.0, 0.0, 0.5, 0.2, 0.1, 0.3]
    actual = _run_cli(
        ["--query-vector", "0.9,0.1,0.0,0.0,0.5,0.2,0.1,0.3", "--top-k", "20", "--max-price", "80.0"]
    )
    expected = _expected(query, top_k=20, max_price=80.0)
    assert expected["hits"], "Test fixture sanity: expected at least one surviving hit for this query."
    _assert_hits_match(actual, expected)
    _assert_departments_match(actual, expected)


def test_category_filter_with_apostrophe():
    query = [0.2, 0.4, 0.4, 0.1, 0.0, 0.3, 0.6, 0.1]
    actual = _run_cli(
        [
            "--query-vector",
            "0.2,0.4,0.4,0.1,0.0,0.3,0.6,0.1",
            "--top-k",
            "20",
            "--max-price",
            "200.0",
            "--category",
            "Women's Apparel",
        ]
    )
    expected = _expected(query, top_k=20, max_price=200.0, category="Women's Apparel")
    assert expected["hits"], "Test fixture sanity: expected at least one Women's Apparel hit for this query."
    for h in actual["hits"]:
        assert h["category"] == "Women's Apparel", (
            f"Category filter leaked a non-matching category: {h['category']}."
        )
    _assert_hits_match(actual, expected)
    _assert_departments_match(actual, expected)


def test_empty_result_edge_case():
    query = [0.9, 0.1, 0.0, 0.0, 0.5, 0.2, 0.1, 0.3]
    actual = _run_cli(
        ["--query-vector", "0.9,0.1,0.0,0.0,0.5,0.2,0.1,0.3", "--top-k", "5", "--max-price", "0.0"]
    )
    assert actual["hits"] == [], f"Expected no hits when max-price is 0.0, got {actual['hits']}."
    assert actual["departments"] == [], (
        f"Expected no departments when there are no hits, got {actual['departments']}."
    )
