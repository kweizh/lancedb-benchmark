import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_FILE = os.path.join(PROJECT_DIR, "solution.py")
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid and rid.strip(), "ZEALT_RUN_ID must be set for verification."
    return rid.strip()


def _table_name() -> str:
    return f"products_{_run_id()}"


def _run_solution():
    assert os.path.isfile(SOLUTION_FILE), (
        f"Candidate solution module not found at {SOLUTION_FILE}."
    )
    if os.path.isdir(LANCEDB_DIR):
        shutil.rmtree(LANCEDB_DIR)
    proc = subprocess.run(
        [sys.executable, SOLUTION_FILE],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        "Running `python3 solution.py` failed.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


@pytest.fixture(scope="session", autouse=True)
def build_solution():
    _run_solution()
    yield


def _open_table():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    name = _table_name()
    names = list(db.table_names())
    assert name in names, (
        f"Expected table '{name}' in LanceDB at {LANCEDB_DIR}; found tables: {names}."
    )
    return db.open_table(name)


def _read_all_rows():
    tbl = _open_table()
    df = tbl.to_pandas()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "id": int(r["id"]),
                "name": str(r["name"]),
                "brand": str(r["brand"]),
                "category": str(r["category"]),
                "price": float(r["price"]),
                "banned_country": str(r["banned_country"]),
                "vector": np.asarray(r["vector"], dtype=np.float32),
            }
        )
    return rows


def _l2_distance(a, b):
    da = np.asarray(a, dtype=np.float32)
    db = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(da - db))


def _ground_truth(rows, predicate, query_vec, k):
    filtered = [r for r in rows if predicate(r)]
    scored = [(r, _l2_distance(query_vec, r["vector"])) for r in filtered]
    scored.sort(key=lambda x: (x[1], x[0]["id"]))
    return [r for (r, _d) in scored[:k]]


def _import_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    import solution  # type: ignore

    return solution


def _validate_result_shape(result, k):
    assert isinstance(result, list), (
        f"Search function must return a list, got {type(result).__name__}."
    )
    assert len(result) <= k, (
        f"Search function returned more than k={k} rows: got {len(result)} rows."
    )
    for row in result:
        assert isinstance(row, dict), (
            f"Each result entry must be a dict, got {type(row).__name__}."
        )
        assert set(row.keys()) == {"id", "name", "brand", "category", "price"}, (
            "Each result dict must have exactly keys "
            "{id, name, brand, category, price}, got: " + str(sorted(row.keys()))
        )


def test_schema_and_row_count():
    import pyarrow as pa

    tbl = _open_table()
    schema = tbl.schema
    expected_fields = {
        "id": pa.int64(),
        "name": pa.utf8(),
        "brand": pa.utf8(),
        "category": pa.utf8(),
        "price": pa.float64(),
        "banned_country": pa.utf8(),
    }
    field_names = [f.name for f in schema]
    for col, expected_type in expected_fields.items():
        assert col in field_names, (
            f"Missing column '{col}' in schema; got fields: {field_names}."
        )
        actual_type = schema.field(col).type
        assert actual_type == expected_type, (
            f"Column '{col}' has type {actual_type}, expected {expected_type}."
        )

    assert "vector" in field_names, (
        f"Missing 'vector' column in schema; got fields: {field_names}."
    )
    vec_type = schema.field("vector").type
    assert pa.types.is_fixed_size_list(vec_type), (
        f"Expected 'vector' to be a fixed_size_list, got {vec_type}."
    )
    assert vec_type.list_size == 16, (
        f"Expected fixed_size_list of length 16 for 'vector', got {vec_type.list_size}."
    )
    assert (
        pa.types.is_floating(vec_type.value_type)
        and vec_type.value_type.bit_width == 32
    ), f"Expected 'vector' value type to be float32, got {vec_type.value_type}."

    row_count = tbl.count_rows()
    assert row_count == 300, (
        f"Expected exactly 300 rows in '{_table_name()}', got {row_count}."
    )


def test_brand_and_category_cardinality():
    rows = _read_all_rows()
    brands = {r["brand"] for r in rows}
    categories = {r["category"] for r in rows}
    expected_brands = {f"Brand{c}" for c in "ABCDEFGHIJ"}
    expected_categories = {f"Cat{i}" for i in range(1, 7)}
    assert expected_brands.issubset(brands), (
        f"Expected all 10 brands {sorted(expected_brands)} to appear; got {sorted(brands)}."
    )
    assert expected_categories.issubset(categories), (
        f"Expected all 6 categories {sorted(expected_categories)} to appear; "
        f"got {sorted(categories)}."
    )


def test_fixture_is_deterministic_across_reruns():
    rows_first = _read_all_rows()
    # Re-run solution.py from scratch.
    _run_solution()
    rows_second = _read_all_rows()

    assert len(rows_first) == len(rows_second), (
        "Re-running solution.py changed the row count; expected identical fixture."
    )

    by_id_a = {r["id"]: r for r in rows_first}
    by_id_b = {r["id"]: r for r in rows_second}
    assert set(by_id_a.keys()) == set(by_id_b.keys()), (
        "Re-running solution.py changed the set of ids in the table."
    )
    for rid, a in by_id_a.items():
        b = by_id_b[rid]
        for col in ("name", "brand", "category", "banned_country"):
            assert a[col] == b[col], (
                f"Re-running solution.py changed column '{col}' for id={rid}: "
                f"{a[col]!r} -> {b[col]!r}."
            )
        assert a["price"] == pytest.approx(b["price"], abs=1e-9), (
            f"Re-running solution.py changed price for id={rid}: "
            f"{a['price']} -> {b['price']}."
        )
        assert np.array_equal(a["vector"], b["vector"]), (
            f"Re-running solution.py changed the vector for id={rid}."
        )


Q1 = [0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5]
Q2 = [0.5] * 16
Q3 = [-0.25] * 16
Q4 = [1.0, -1.0] * 8


def test_exclude_brands_correctness():
    solution = _import_solution()
    rows = _read_all_rows()
    banned = ["BrandA", "BrandC", "BrandE"]
    result = solution.exclude_brands(Q1, banned, k=10)
    _validate_result_shape(result, 10)

    for row in result:
        assert row["brand"] not in banned, (
            f"exclude_brands returned a row with banned brand: {row}."
        )

    gt = _ground_truth(rows, lambda r: r["brand"] not in banned, Q1, 10)
    expected_ids = [r["id"] for r in gt]
    actual_ids = [row["id"] for row in result]
    assert actual_ids == expected_ids, (
        f"exclude_brands top-{len(expected_ids)} id order does not match "
        f"ground truth.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )


def test_exclude_categories_and_price_correctness():
    solution = _import_solution()
    rows = _read_all_rows()
    cat = "Cat2"
    max_price = 250.0
    result = solution.exclude_categories_and_price(Q2, cat, max_price, k=10)
    _validate_result_shape(result, 10)

    for row in result:
        assert row["category"] != cat, (
            f"exclude_categories_and_price returned excluded category: {row}."
        )
        assert row["price"] <= max_price + 1e-9, (
            f"exclude_categories_and_price returned price>{max_price}: {row}."
        )

    gt = _ground_truth(
        rows,
        lambda r: r["category"] != cat and r["price"] <= max_price,
        Q2,
        10,
    )
    expected_ids = [r["id"] for r in gt]
    actual_ids = [row["id"] for row in result]
    assert actual_ids == expected_ids, (
        f"exclude_categories_and_price top-{len(expected_ids)} id order does "
        f"not match ground truth.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )


def test_not_like_search_correctness():
    solution = _import_solution()
    rows = _read_all_rows()
    prefix = "premium"
    result = solution.not_like_search(Q3, prefix, k=10)
    _validate_result_shape(result, 10)

    for row in result:
        assert not row["name"].startswith(prefix), (
            f"not_like_search returned row whose name starts with '{prefix}': {row}."
        )

    gt = _ground_truth(
        rows, lambda r: not r["name"].startswith(prefix), Q3, 10
    )
    expected_ids = [r["id"] for r in gt]
    actual_ids = [row["id"] for row in result]
    assert actual_ids == expected_ids, (
        f"not_like_search top-{len(expected_ids)} id order does not match "
        f"ground truth.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )


def test_complex_negation_correctness():
    solution = _import_solution()
    rows = _read_all_rows()
    result = solution.complex_negation(Q4, k=10)
    _validate_result_shape(result, 10)

    def predicate(r):
        return not (
            r["brand"] == "BrandA"
            or (r["category"] == "Cat3" and r["price"] > 100.0)
        )

    for row in result:
        assert predicate(
            {
                "brand": row["brand"],
                "category": row["category"],
                "price": row["price"],
            }
        ), f"complex_negation returned row violating predicate: {row}."

    gt = _ground_truth(rows, predicate, Q4, 10)
    expected_ids = [r["id"] for r in gt]
    actual_ids = [row["id"] for row in result]
    assert actual_ids == expected_ids, (
        f"complex_negation top-{len(expected_ids)} id order does not match "
        f"ground truth.\nExpected: {expected_ids}\nActual:   {actual_ids}"
    )


def test_exclude_brands_repeat_call_is_stable():
    solution = _import_solution()
    banned = ["BrandA", "BrandC", "BrandE"]
    r1 = solution.exclude_brands(Q1, banned, k=10)
    r2 = solution.exclude_brands(Q1, banned, k=10)
    assert r1 == r2, (
        "exclude_brands returned different results on two consecutive calls "
        "with identical arguments.\nFirst:  "
        f"{[row['id'] for row in r1]}\nSecond: {[row['id'] for row in r2]}"
    )
