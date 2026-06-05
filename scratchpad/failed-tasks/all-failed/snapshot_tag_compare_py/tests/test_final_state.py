import importlib
import importlib.util
import os
import sys

import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = "/app/db"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
EXPECTED_TAGS = {"v1_baseline", "v2_extended", "v3_pruned"}


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID must be set in the verifier environment."
    return f"documents_{run_id}"


def _open_table():
    import lancedb

    db = lancedb.connect(DB_DIR)
    name = _table_name()
    list_fn = getattr(db, "list_tables", None) or db.table_names
    names = list(list_fn())
    assert name in names, (
        f"LanceDB table {name!r} not found in {DB_DIR}. Found tables: {names}."
    )
    return db.open_table(name)


def _load_solution_module():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Candidate solution module not found at {SOLUTION_PATH}."
    )
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = module
    spec.loader.exec_module(module)
    return module


def _ids_at_tag(tag):
    table = _open_table()
    table.checkout(tag)
    try:
        arrow = table.to_arrow()
    finally:
        table.checkout_latest()
    cols = arrow.schema.names
    assert "id" in cols, f"Tagged snapshot {tag!r} is missing the `id` column. Schema: {cols}."
    return sorted(int(x) for x in arrow.column("id").to_pylist())


def test_table_exists_and_has_65_rows():
    table = _open_table()
    count = table.count_rows()
    assert count == 65, f"Expected 65 rows in the latest version, got {count}."


def test_three_tags_present_with_monotonic_versions():
    table = _open_table()
    assert hasattr(table, "tags"), (
        "table.tags accessor not present — install a LanceDB version that exposes the tags API."
    )
    tags = table.tags.list()
    assert isinstance(tags, dict), f"table.tags.list() must return a dict, got {type(tags)!r}."
    assert set(tags.keys()) == EXPECTED_TAGS, (
        f"Expected exactly the tags {sorted(EXPECTED_TAGS)}, got {sorted(tags.keys())}."
    )

    def _version_of(name):
        entry = tags[name]
        if isinstance(entry, dict):
            assert "version" in entry, f"Tag entry for {name!r} missing 'version' key: {entry!r}."
            return int(entry["version"])
        return int(entry)

    v1 = _version_of("v1_baseline")
    v2 = _version_of("v2_extended")
    v3 = _version_of("v3_pruned")
    assert v1 < v2 < v3, (
        f"Tag versions must be strictly increasing in creation order: "
        f"v1_baseline={v1}, v2_extended={v2}, v3_pruned={v3}."
    )


def test_v1_baseline_snapshot_ids():
    ids = _ids_at_tag("v1_baseline")
    assert ids == list(range(0, 50)), (
        f"v1_baseline must contain ids 0..49 (50 rows); got first 5={ids[:5]}, len={len(ids)}."
    )


def test_v2_extended_snapshot_ids():
    ids = _ids_at_tag("v2_extended")
    assert ids == list(range(0, 70)), (
        f"v2_extended must contain ids 0..69 (70 rows); got len={len(ids)}, last 5={ids[-5:]}."
    )


def test_v3_pruned_snapshot_ids():
    ids = _ids_at_tag("v3_pruned")
    assert ids == list(range(5, 70)), (
        f"v3_pruned must contain ids 5..69 (65 rows); got len={len(ids)}, first 5={ids[:5]}."
    )


def test_diff_v1_to_v2_returns_twenty_added():
    solution = _load_solution_module()
    assert hasattr(solution, "diff"), "solution.py must expose a top-level `diff` function."
    result = solution.diff(DB_DIR, _table_name(), "v1_baseline", "v2_extended")
    assert isinstance(result, dict), f"diff(...) must return a dict, got {type(result)!r}."
    assert set(result.keys()) == {"added_ids", "removed_ids", "common_count"}, (
        f"diff(...) keys must be exactly {{'added_ids','removed_ids','common_count'}}; got {sorted(result.keys())}."
    )
    assert result["added_ids"] == list(range(50, 70)), (
        f"diff(v1_baseline, v2_extended) added_ids must be sorted [50..69]; got {result['added_ids']}."
    )
    assert result["removed_ids"] == [], (
        f"diff(v1_baseline, v2_extended) removed_ids must be empty; got {result['removed_ids']}."
    )
    assert result["common_count"] == 50, (
        f"diff(v1_baseline, v2_extended) common_count must be 50; got {result['common_count']}."
    )


def test_diff_v2_to_v3_returns_five_removed():
    solution = _load_solution_module()
    assert hasattr(solution, "diff"), "solution.py must expose a top-level `diff` function."
    result = solution.diff(DB_DIR, _table_name(), "v2_extended", "v3_pruned")
    assert isinstance(result, dict), f"diff(...) must return a dict, got {type(result)!r}."
    assert set(result.keys()) == {"added_ids", "removed_ids", "common_count"}, (
        f"diff(...) keys must be exactly {{'added_ids','removed_ids','common_count'}}; got {sorted(result.keys())}."
    )
    assert result["added_ids"] == [], (
        f"diff(v2_extended, v3_pruned) added_ids must be empty; got {result['added_ids']}."
    )
    assert result["removed_ids"] == [0, 1, 2, 3, 4], (
        f"diff(v2_extended, v3_pruned) removed_ids must be [0,1,2,3,4]; got {result['removed_ids']}."
    )
    assert result["common_count"] == 65, (
        f"diff(v2_extended, v3_pruned) common_count must be 65; got {result['common_count']}."
    )


def test_diff_restores_latest_version():
    # After calling diff, the candidate must have called table.checkout_latest() so the
    # live table is observable to subsequent callers.
    solution = _load_solution_module()
    solution.diff(DB_DIR, _table_name(), "v1_baseline", "v3_pruned")
    table = _open_table()
    assert table.count_rows() == 65, (
        "After diff(), reopening the table must show the latest version with 65 rows."
    )
