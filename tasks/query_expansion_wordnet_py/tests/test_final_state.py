import importlib
import os
import re
import sys

import pytest


PROJECT_DIR = "/home/user/myproject"
LANCEDB_URI = "/app/lancedb_data"
LANCEDB_TABLE = "docs"


def _open_table():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    return db.open_table(LANCEDB_TABLE)


def _synonym_only_ids():
    """Docs whose lowercased word-tokens contain 'automobile' or 'vehicle' but
    do NOT contain 'car'. Computed live from the seeded fixture so the verifier
    never assumes a particular id assignment."""
    tbl = _open_table()
    df = tbl.to_pandas()
    out = set()
    for _, row in df.iterrows():
        tokens = set(re.findall(r"\b\w+\b", str(row["content"]).lower()))
        if ("automobile" in tokens or "vehicle" in tokens) and "car" not in tokens:
            out.add(int(row["id"]))
    return out


@pytest.fixture(scope="module")
def solution_module():
    """Import the candidate's solution.py module once per test module."""
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    sys.modules.pop("solution", None)
    mod = importlib.import_module("solution")
    return mod


def test_solution_file_exists():
    path = os.path.join(PROJECT_DIR, "solution.py")
    assert os.path.isfile(path), f"Expected candidate module at {path}."


def test_expanded_search_is_callable(solution_module):
    assert hasattr(solution_module, "expanded_search"), (
        "solution.py does not expose `expanded_search`."
    )
    assert callable(solution_module.expanded_search), (
        "solution.expanded_search is not callable."
    )


def test_expanded_search_return_contract(solution_module):
    result = solution_module.expanded_search("car", 10)
    assert isinstance(result, list), (
        f"expanded_search must return list, got {type(result).__name__}: {result!r}"
    )
    assert 0 < len(result) <= 10, (
        f"expanded_search result length must be in (0, 10]; got {len(result)}: {result!r}"
    )
    for v in result:
        assert isinstance(v, int) and not isinstance(v, bool), (
            f"expanded_search result must contain plain ints, got element "
            f"{v!r} of type {type(v).__name__}. Full list: {result!r}"
        )


def test_native_fts_index_on_content(solution_module):
    """After the first invocation, a native (non-Tantivy) FTS index must exist
    on the `content` column."""
    solution_module.expanded_search("car", 10)
    tbl = _open_table()
    indices = list(tbl.list_indices())
    fts_on_content = [
        idx for idx in indices
        if "content" in (getattr(idx, "columns", None) or [])
    ]
    assert fts_on_content, (
        f"Expected an FTS index on the 'content' column after expanded_search; "
        f"list_indices() returned: "
        f"{[(getattr(i, 'name', None), getattr(i, 'columns', None), getattr(i, 'index_type', None)) for i in indices]}"
    )
    for idx in fts_on_content:
        itype = str(getattr(idx, "index_type", "")).lower()
        assert "tantivy" not in itype, (
            f"FTS index on 'content' must be Lance-native (use_tantivy=False); "
            f"got index_type={itype!r}"
        )
        assert "fts" in itype or "inverted" in itype, (
            f"Expected index_type to look like FTS/Inverted; got {itype!r}"
        )


def test_fixture_invariant_synonym_only_docs_exist():
    """Sanity-check the fixture: there must be at least 3 docs containing
    'automobile'/'vehicle' but not 'car', otherwise the recall test is vacuous."""
    syn_ids = _synonym_only_ids()
    assert len(syn_ids) >= 3, (
        f"Fixture invariant violated: expected ≥3 synonym-only docs; found {syn_ids}"
    )


def test_baseline_isolation(solution_module):
    """The plain FTS baseline for 'car' must NOT surface any synonym-only doc."""
    solution_module.expanded_search("car", 10)  # ensure index exists
    tbl = _open_table()
    baseline = [
        int(r["id"])
        for r in tbl.search("car", query_type="fts").limit(10).to_list()
    ]
    syn_ids = _synonym_only_ids()
    leakage = set(baseline) & syn_ids
    assert leakage == set(), (
        "Baseline FTS for 'car' unexpectedly returned synonym-only docs; "
        f"baseline={baseline}, synonym_only_ids={syn_ids}, leakage={leakage}"
    )


def test_synonym_recall_via_expansion(solution_module):
    """expanded_search must recover ≥3 synonym-only docs in its top-10."""
    syn_ids = _synonym_only_ids()
    result = solution_module.expanded_search("car", 10)
    overlap = set(result) & syn_ids
    assert len(overlap) >= 3, (
        f"Expected expanded_search('car', 10) to surface ≥3 synonym-only docs; "
        f"got result={result}, synonym_only_ids={syn_ids}, overlap={overlap}."
    )


def test_expansion_differs_from_baseline(solution_module):
    """The two pipelines must demonstrably differ on the rigged query."""
    tbl = _open_table()
    baseline = [
        int(r["id"])
        for r in tbl.search("car", query_type="fts").limit(10).to_list()
    ]
    result = solution_module.expanded_search("car", 10)
    assert set(result) != set(baseline), (
        f"expanded_search produced the same result set as the plain FTS baseline; "
        f"result={result}, baseline={baseline}"
    )


def test_expanded_search_is_idempotent(solution_module):
    """Calling expanded_search twice must return the same ordered list."""
    r1 = solution_module.expanded_search("car", 10)
    r2 = solution_module.expanded_search("car", 10)
    assert r1 == r2, (
        f"expanded_search is not deterministic across calls; first={r1}, second={r2}"
    )
