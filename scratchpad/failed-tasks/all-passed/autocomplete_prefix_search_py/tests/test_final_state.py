import importlib
import importlib.util
import os
import sys

import lancedb
import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"


def _load_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location(
        "solution_under_test", os.path.join(PROJECT_DIR, "solution.py")
    )
    assert spec is not None and spec.loader is not None, (
        f"Could not load solution module from {PROJECT_DIR}/solution.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def solution_mod():
    return _load_solution()


@pytest.fixture(scope="session")
def lance_handles():
    uri = os.environ["LANCE_DB_URI"]
    movies_name = os.environ["MOVIES_TABLE"]
    prefix_name = os.environ["PREFIX_TABLE"]
    db = lancedb.connect(uri)
    return {
        "db": db,
        "movies": db.open_table(movies_name),
        "prefix": db.open_table(prefix_name),
        "movies_name": movies_name,
        "prefix_name": prefix_name,
    }


def _prefix_lookup_vector(prefix_tbl, prefix_lower: str) -> np.ndarray:
    safe = prefix_lower.replace("'", "''")
    df = prefix_tbl.search().where(f"prefix = '{safe}'").limit(1).to_pandas()
    assert len(df) == 1, (
        f"Expected exactly one row in prefix_vectors for prefix='{prefix_lower}', got {len(df)}"
    )
    vec = np.asarray(df.iloc[0]["vector"], dtype=np.float32)
    assert vec.shape == (32,), (
        f"Expected 32-d vector for prefix '{prefix_lower}', got shape {vec.shape}"
    )
    return vec


def _prefix_matches(movies_tbl, prefix_lower: str):
    """All movie rows whose lower(title) starts with the given lowercase prefix."""
    safe = prefix_lower.replace("'", "''")
    df = (
        movies_tbl.search()
        .where(f"title_lower LIKE '{safe}%'")
        .limit(10_000)
        .to_pandas()
    )
    return df


def _result_keys_ok(result):
    assert isinstance(result, list), f"autocomplete must return a list, got {type(result)}"
    for item in result:
        assert isinstance(item, dict), f"Each result must be a dict, got {type(item)}"
        for key in ("id", "title", "popularity", "source"):
            assert key in item, f"Each result dict must contain key '{key}'. Got keys: {sorted(item.keys())}"
        assert item["source"] in ("prefix", "semantic"), (
            f"source must be 'prefix' or 'semantic', got {item['source']!r}"
        )


def test_prefix_only_path_many_matches(solution_mod, lance_handles):
    """`autocomplete('Crystal', 10)` should return the top-10 movies by popularity DESC,
    all source='prefix', all titles starting with 'crystal' (case-insensitive)."""
    movies = lance_handles["movies"]
    df = _prefix_matches(movies, "crystal")
    assert len(df) >= 10, (
        f"Seed fixture must provide at least 10 movies whose lower(title) starts with 'crystal'; "
        f"found {len(df)}"
    )

    expected_sorted = df.sort_values(
        by=["popularity", "id"], ascending=[False, True]
    )
    expected_top_ids = list(expected_sorted["id"].iloc[:10].astype(int))

    result = solution_mod.autocomplete("Crystal", 10)
    _result_keys_ok(result)
    assert len(result) == 10, f"Expected 10 results, got {len(result)}"

    for item in result:
        assert item["source"] == "prefix", (
            f"All entries must have source='prefix' when prefix yields >= k matches; "
            f"got source={item['source']!r} for id={item['id']}"
        )
        assert str(item["title"]).lower().startswith("crystal"), (
            f"Returned title {item['title']!r} does not start with 'crystal'"
        )

    pops = [float(it["popularity"]) for it in result]
    for a, b in zip(pops, pops[1:]):
        assert a >= b, f"popularity must be non-increasing; got {pops}"

    returned_ids = set(int(it["id"]) for it in result)
    assert returned_ids == set(expected_top_ids), (
        f"Returned id set {sorted(returned_ids)} does not match expected top-10 by popularity {sorted(expected_top_ids)}"
    )


def test_fallback_path_partial_matches(solution_mod, lance_handles):
    """`autocomplete('Zephyr', 10)` should return 2 prefix matches + 8 semantic via prefix_vectors lookup."""
    movies = lance_handles["movies"]
    prefix_tbl = lance_handles["prefix"]

    df = _prefix_matches(movies, "zephyr")
    assert len(df) == 2, (
        f"Seed fixture must provide exactly 2 movies whose lower(title) starts with 'zephyr'; got {len(df)}"
    )

    prefix_expected_ids_ordered = list(
        df.sort_values(by=["popularity", "id"], ascending=[False, True])["id"]
        .astype(int)
    )

    qvec = _prefix_lookup_vector(prefix_tbl, "zephyr")
    exclude_clause = "id NOT IN (" + ", ".join(str(i) for i in prefix_expected_ids_ordered) + ")"
    semantic_df = (
        movies.search(qvec)
        .where(exclude_clause, prefilter=True)
        .limit(8)
        .to_pandas()
    )
    expected_semantic_ids = set(int(i) for i in semantic_df["id"].tolist())
    assert len(expected_semantic_ids) == 8, (
        f"Expected exactly 8 semantic ids, got {len(expected_semantic_ids)}"
    )

    result = solution_mod.autocomplete("Zephyr", 10)
    _result_keys_ok(result)
    assert len(result) == 10, f"Expected 10 results, got {len(result)}"

    prefix_part = result[:2]
    semantic_part = result[2:]

    for item in prefix_part:
        assert item["source"] == "prefix", (
            f"First M items must have source='prefix'; got {item['source']!r}"
        )
        assert str(item["title"]).lower().startswith("zephyr"), (
            f"Prefix-part title {item['title']!r} must start with 'zephyr'"
        )
    pops = [float(it["popularity"]) for it in prefix_part]
    for a, b in zip(pops, pops[1:]):
        assert a >= b, f"prefix portion popularity must be non-increasing; got {pops}"
    assert [int(it["id"]) for it in prefix_part] == prefix_expected_ids_ordered, (
        f"Prefix portion ids/order {[int(it['id']) for it in prefix_part]} != expected {prefix_expected_ids_ordered}"
    )

    for item in semantic_part:
        assert item["source"] == "semantic", (
            f"Tail items must have source='semantic'; got {item['source']!r}"
        )
        assert int(item["id"]) not in set(prefix_expected_ids_ordered), (
            f"Semantic item id={item['id']} must not duplicate prefix portion"
        )
    semantic_ids = set(int(it["id"]) for it in semantic_part)
    assert semantic_ids == expected_semantic_ids, (
        f"Semantic id set {sorted(semantic_ids)} != expected {sorted(expected_semantic_ids)}"
    )


def test_pure_semantic_path_zero_prefix_matches(solution_mod, lance_handles):
    """`autocomplete('orbital', 10)`: 0 prefix matches, all results from vector search."""
    movies = lance_handles["movies"]
    prefix_tbl = lance_handles["prefix"]

    df = _prefix_matches(movies, "orbital")
    assert len(df) == 0, (
        f"Seed fixture must provide ZERO movies whose lower(title) starts with 'orbital'; got {len(df)}"
    )

    qvec = _prefix_lookup_vector(prefix_tbl, "orbital")
    semantic_df = movies.search(qvec).limit(10).to_pandas()
    expected_ids = set(int(i) for i in semantic_df["id"].tolist())
    assert len(expected_ids) == 10

    result = solution_mod.autocomplete("orbital", 10)
    _result_keys_ok(result)
    assert len(result) == 10, f"Expected 10 results, got {len(result)}"
    for item in result:
        assert item["source"] == "semantic", (
            f"All entries must have source='semantic' when no prefix match exists; got {item['source']!r}"
        )
    got_ids = set(int(it["id"]) for it in result)
    assert got_ids == expected_ids, (
        f"Semantic id set {sorted(got_ids)} != expected {sorted(expected_ids)}"
    )


def test_case_insensitive_prefix(solution_mod):
    """`autocomplete('crystal', 10)` and `autocomplete('CRYSTAL', 10)` must return the same id sequence as 'Crystal'."""
    base = [int(it["id"]) for it in solution_mod.autocomplete("Crystal", 10)]
    lower = [int(it["id"]) for it in solution_mod.autocomplete("crystal", 10)]
    upper = [int(it["id"]) for it in solution_mod.autocomplete("CRYSTAL", 10)]
    assert lower == base, f"Lowercase prefix returned {lower}, expected {base}"
    assert upper == base, f"Uppercase prefix returned {upper}, expected {base}"


def test_prefix_table_schema_and_count(lance_handles):
    """Seed `prefix_vectors` table must have 50 rows and a 32-d float vector column."""
    prefix_tbl = lance_handles["prefix"]
    assert prefix_tbl.count_rows() == 50, (
        f"prefix_vectors row count should be 50, got {prefix_tbl.count_rows()}"
    )
    schema = prefix_tbl.schema
    field_names = set(schema.names)
    assert "prefix" in field_names, f"prefix_vectors must have a 'prefix' column. Got {field_names}"
    assert "vector" in field_names, f"prefix_vectors must have a 'vector' column. Got {field_names}"
    sample = prefix_tbl.search().limit(1).to_pandas()
    vec = np.asarray(sample.iloc[0]["vector"], dtype=np.float32)
    assert vec.shape == (32,), f"prefix_vectors.vector must be 32-d, got {vec.shape}"
