import json
import math
import os
import subprocess

import duckdb
import lancedb
import numpy as np
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CATEGORIES = ["books", "electronics", "clothing", "food", "toys"]


def _run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID environment variable must be set."
    return rid


def _table_name():
    return f"products_{_run_id()}"


@pytest.fixture(scope="session")
def build_and_summary():
    # Setup: clean leftover artifacts.
    summary_path = os.path.join(PROJECT_DIR, "summary.json")
    if os.path.isfile(summary_path):
        os.remove(summary_path)
    if os.path.isdir(DATA_DIR):
        import shutil
        shutil.rmtree(DATA_DIR)

    # Run build.
    build = subprocess.run(
        ["python3", "run.py", "build"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert build.returncode == 0, (
        f"`python3 run.py build` failed with code {build.returncode}. "
        f"stdout={build.stdout!r} stderr={build.stderr!r}"
    )

    # Run summary.
    summary = subprocess.run(
        ["python3", "run.py", "summary"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert summary.returncode == 0, (
        f"`python3 run.py summary` failed with code {summary.returncode}. "
        f"stdout={summary.stdout!r} stderr={summary.stderr!r}"
    )

    # Parse stdout as JSON.
    try:
        data = json.loads(summary.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"`python3 run.py summary` stdout is not valid JSON: {e}. "
            f"stdout={summary.stdout!r}"
        )
    return data


def test_summary_has_exactly_five_known_categories(build_and_summary):
    keys = set(build_and_summary.keys())
    assert keys == set(CATEGORIES), (
        f"Summary must contain exactly these category keys: {sorted(CATEGORIES)}. Got: {sorted(keys)}"
    )


def test_summary_entry_shapes_and_ranges(build_and_summary):
    for cat, entry in build_and_summary.items():
        assert isinstance(entry, dict), f"Category '{cat}' value must be a dict, got {type(entry).__name__}."
        for key in ("count", "avg_price", "in_stock_rate"):
            assert key in entry, f"Category '{cat}' entry is missing required key '{key}'."
        assert isinstance(entry["count"], int) and entry["count"] > 0, (
            f"Category '{cat}' count must be a positive int, got {entry['count']!r}."
        )
        assert isinstance(entry["avg_price"], (int, float)) and entry["avg_price"] > 0, (
            f"Category '{cat}' avg_price must be a positive number, got {entry['avg_price']!r}."
        )
        rate = entry["in_stock_rate"]
        assert isinstance(rate, (int, float)) and 0.0 <= float(rate) <= 1.0, (
            f"Category '{cat}' in_stock_rate must be in [0.0, 1.0], got {rate!r}."
        )


def test_counts_sum_to_1000(build_and_summary):
    total = sum(entry["count"] for entry in build_and_summary.values())
    assert total == 1000, f"Sum of per-category counts must equal 1000, got {total}."


def _regenerate_ground_truth():
    rng = np.random.default_rng(2026)
    category_indices = rng.integers(0, 5, size=1000)
    prices = rng.uniform(1.0, 1000.0, size=1000)
    in_stock_mask = rng.random(1000) < 0.7
    # Consume the embedding draws so the RNG state ordering is exercised; not used in aggregation.
    _ = rng.standard_normal((1000, 16)).astype(np.float32)

    truth = {}
    for idx, cat in enumerate(CATEGORIES):
        mask = category_indices == idx
        n = int(mask.sum())
        avg_price = float(prices[mask].mean())
        in_stock_rate = float(in_stock_mask[mask].mean())
        truth[cat] = {"count": n, "avg_price": avg_price, "in_stock_rate": in_stock_rate}
    return truth


def test_summary_matches_regenerated_ground_truth(build_and_summary):
    truth = _regenerate_ground_truth()
    for cat in CATEGORIES:
        got = build_and_summary[cat]
        want = truth[cat]
        assert got["count"] == want["count"], (
            f"Category '{cat}' count mismatch: candidate={got['count']} truth={want['count']}."
        )
        assert math.isclose(got["avg_price"], want["avg_price"], rel_tol=1e-6, abs_tol=1e-6), (
            f"Category '{cat}' avg_price mismatch: candidate={got['avg_price']} truth={want['avg_price']}."
        )
        assert math.isclose(got["in_stock_rate"], want["in_stock_rate"], rel_tol=1e-6, abs_tol=1e-6), (
            f"Category '{cat}' in_stock_rate mismatch: candidate={got['in_stock_rate']} truth={want['in_stock_rate']}."
        )


def test_table_persisted_with_correct_shape(build_and_summary):
    db = lancedb.connect(DATA_DIR)
    name = _table_name()
    assert name in db.table_names(), (
        f"Expected LanceDB table '{name}' under {DATA_DIR}, got tables: {db.table_names()}"
    )
    tbl = db.open_table(name)
    assert tbl.count_rows() == 1000, f"LanceDB table must have exactly 1000 rows, got {tbl.count_rows()}."

    schema = tbl.schema
    field = schema.field("embedding")
    ftype = field.type
    assert pa.types.is_fixed_size_list(ftype), (
        f"'embedding' must be a fixed_size_list, got {ftype}."
    )
    assert ftype.list_size == 16, f"'embedding' fixed_size_list size must be 16, got {ftype.list_size}."
    value_type = ftype.value_type
    assert pa.types.is_floating(value_type) and value_type.bit_width == 32, (
        f"'embedding' value type must be float32, got {value_type}."
    )


def test_duckdb_over_lance_dataset_matches_counts(build_and_summary):
    db = lancedb.connect(DATA_DIR)
    tbl = db.open_table(_table_name())
    ds = tbl.to_lance()
    reader = ds.scanner(columns=["category", "price", "in_stock"]).to_reader()
    con = duckdb.connect()
    con.register("products_check", reader)
    rows = con.execute(
        "SELECT category, COUNT(*) AS n FROM products_check GROUP BY category"
    ).fetchall()
    duck_counts = {cat: int(n) for cat, n in rows}
    for cat in CATEGORIES:
        assert cat in duck_counts, (
            f"Independent DuckDB-over-Lance scan is missing category '{cat}'. Got: {sorted(duck_counts)}"
        )
        assert duck_counts[cat] == build_and_summary[cat]["count"], (
            f"Independent DuckDB count mismatch for '{cat}': "
            f"duckdb={duck_counts[cat]} candidate={build_and_summary[cat]['count']}"
        )
    assert sum(duck_counts.values()) == 1000, (
        f"Independent DuckDB total row count must be 1000, got {sum(duck_counts.values())}."
    )
