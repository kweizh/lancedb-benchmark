import os
import sys
import importlib

import numpy as np
import pyarrow as pa
import lancedb
import pytest


PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")
EMBED_DIM = 24


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID must be set for verification."
    return f"articles_{run_id}"


def _expected_rows_for_year(year: int):
    """Reproduce the exact fixture rows generated at build time for a given year partition."""
    rng = np.random.default_rng(2026 + (year - 2022))
    base_id = (year - 2022) * 200
    rows = []
    for i in range(200):
        emb = rng.standard_normal(EMBED_DIM).astype(np.float32)
        rows.append({
            "id": base_id + i,
            "title": f"doc-{year}-{i:03d}",
            "embedding": emb,
            "year": year,
        })
    return rows


@pytest.fixture(scope="session")
def loaded_solution():
    """Import the candidate's solution module, executing its ingest side effects."""
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Force a fresh import in case anything was cached.
    if "solution" in sys.modules:
        del sys.modules["solution"]
    mod = importlib.import_module("solution")
    return mod


@pytest.fixture(scope="session")
def lance_table(loaded_solution):
    db = lancedb.connect(LANCEDB_DIR)
    name = _table_name()
    assert name in db.table_names(), (
        f"Expected destination table {name!r} in lancedb at {LANCEDB_DIR}; "
        f"found tables: {db.table_names()}"
    )
    return db.open_table(name)


def test_total_row_count(lance_table):
    n = lance_table.count_rows()
    assert n == 600, f"Expected 600 total rows, got {n}."


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_row_count_per_year(lance_table, year):
    n = lance_table.count_rows(filter=f"year = {year}")
    assert n == 200, f"Expected 200 rows for year={year}, got {n}."


def test_schema_has_required_columns(lance_table):
    schema = lance_table.schema
    names = set(schema.names)
    for col in ("id", "title", "embedding", "year"):
        assert col in names, f"Destination table missing required column `{col}`."

    # embedding must be a fixed-size list of float32, list size 24.
    embed_field = schema.field("embedding")
    embed_type = embed_field.type
    assert pa.types.is_fixed_size_list(embed_type), (
        f"`embedding` must be a fixed_size_list, got {embed_type!r}."
    )
    assert embed_type.list_size == EMBED_DIM, (
        f"`embedding` list size must be {EMBED_DIM}, got {embed_type.list_size}."
    )
    value_type = embed_type.value_type
    assert pa.types.is_float32(value_type), (
        f"`embedding` value type must be float32, got {value_type!r}."
    )

    # year must be an integer type.
    year_type = schema.field("year").type
    assert pa.types.is_integer(year_type), (
        f"`year` column must be an integer type, got {year_type!r}."
    )


def test_year_values_are_correct_set(lance_table):
    df = lance_table.to_pandas()
    distinct = set(int(y) for y in df["year"].unique().tolist())
    assert distinct == {2022, 2023, 2024}, (
        f"Distinct years should be {{2022, 2023, 2024}}, got {distinct}."
    )


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_search_year_returns_only_requested_year(loaded_solution, year):
    vec = [0.0] * EMBED_DIM
    res = loaded_solution.search_year(vec, year, k=10)
    assert isinstance(res, list), f"search_year must return a list, got {type(res)!r}."
    assert len(res) <= 10
    assert len(res) > 0, f"Expected at least one result for year={year}."
    for row in res:
        assert isinstance(row, dict), f"Each result must be a dict, got {type(row)!r}."
        assert "year" in row, f"Result missing `year` key: {row!r}."
        assert int(row["year"]) == year, (
            f"search_year(year={year}) returned row with year={row['year']!r}: {row!r}"
        )
        assert "id" in row and "title" in row, (
            f"Result missing required keys (`id`, `title`): {row!r}"
        )


def test_search_year_top1_matches_seeded_embedding(loaded_solution, lance_table):
    """Use a known seeded embedding (year=2024, first row) as the query vector.

    The candidate's search_year MUST return that exact row at rank 1.
    """
    seeded = _expected_rows_for_year(2024)
    target = seeded[0]
    qvec = target["embedding"].tolist()

    res = loaded_solution.search_year(qvec, 2024, k=5)
    assert len(res) >= 1, f"Expected at least 1 result, got {res!r}."
    assert int(res[0]["id"]) == target["id"], (
        f"Top-1 search_year result should match the seeded row id={target['id']}, "
        f"got id={res[0]['id']}."
    )


def test_search_year_matches_direct_lancedb_query(loaded_solution, lance_table):
    """Candidate's search_year ranking MUST equal a direct LanceDB year-filtered vector search."""
    seeded = _expected_rows_for_year(2024)
    # Choose a query vector that is NOT exactly any single row's embedding (mean of first 3).
    qvec = (
        seeded[0]["embedding"]
        + seeded[1]["embedding"]
        + seeded[2]["embedding"]
    ) / 3.0

    candidate = loaded_solution.search_year(qvec.tolist(), 2024, k=5)
    candidate_ids = [int(r["id"]) for r in candidate]

    direct = (
        lance_table.search(qvec.astype(np.float32))
        .where("year = 2024")
        .limit(5)
        .to_list()
    )
    direct_ids = [int(r["id"]) for r in direct]

    assert candidate_ids == direct_ids, (
        f"search_year top-5 must match direct LanceDB query.\n"
        f"  candidate ids: {candidate_ids}\n"
        f"  direct ids:    {direct_ids}"
    )


def test_search_year_returns_full_k_within_partition(loaded_solution):
    """If the candidate were post-filtering an unfiltered small top-K, they would not
    consistently get a full k=5 for the requested year. Since the partition has 200
    rows and we ask for k=5, a correct server-side filter MUST always return 5 rows.
    """
    # Query vector taken from year=2022 partition, so a naive top-5-then-filter would
    # likely surface ~0 year=2023 rows.
    qvec_src = _expected_rows_for_year(2022)[0]["embedding"]
    res = loaded_solution.search_year(qvec_src.tolist(), 2023, k=5)
    assert len(res) == 5, (
        f"search_year(year=2023, k=5) should always return exactly 5 rows "
        f"(partition has 200), got {len(res)}."
    )
    for row in res:
        assert int(row["year"]) == 2023, (
            f"search_year(year=2023) returned wrong-year row: {row!r}"
        )


def test_idempotent_reimport(loaded_solution, lance_table):
    """Re-importing solution must not duplicate rows."""
    if "solution" in sys.modules:
        del sys.modules["solution"]
    importlib.import_module("solution")
    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_table_name())
    assert tbl.count_rows() == 600, (
        f"After re-import, expected 600 rows, got {tbl.count_rows()}; "
        "ingest is not idempotent."
    )
