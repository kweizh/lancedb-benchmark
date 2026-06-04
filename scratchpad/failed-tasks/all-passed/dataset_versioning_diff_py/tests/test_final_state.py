import importlib.util
import json
import math
import os
import sys

import pytest


PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
LANCEDB_PATH = "/data/lancedb"
EXPECTED_PATH = "/opt/expected_diffs.json"


def _load_solution_module():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Solution file {SOLUTION_PATH} does not exist; candidate must create it."
    )
    # Load the candidate solution into a fresh module each call so we never
    # depend on import-cache state between tests.
    spec = importlib.util.spec_from_file_location("candidate_solution", SOLUTION_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solution"] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "diff_versions"), (
        "solution.py must export a callable named 'diff_versions'."
    )
    assert callable(mod.diff_versions), "'diff_versions' must be callable."
    return mod


def _load_expected():
    assert os.path.isfile(EXPECTED_PATH), (
        f"Expected ground-truth file {EXPECTED_PATH} not found; "
        "Docker image was built incorrectly."
    )
    with open(EXPECTED_PATH) as f:
        return json.load(f)


def _normalize_keys(d):
    """Return a dict copy with int(value) for integer-valued keys/lists where needed."""
    return d


def _check_added_removed(actual_list, expected_list, label):
    assert isinstance(actual_list, list), (
        f"Expected diff_versions(...)[{label!r}] to be a list, got {type(actual_list).__name__}."
    )
    actual_ints = sorted(int(x) for x in actual_list)
    expected_ints = sorted(int(x) for x in expected_list)
    assert actual_ints == expected_ints, (
        f"Mismatch for '{label}': expected {expected_ints!r}, got {actual_ints!r}."
    )


def _check_value(field, expected, actual, ctx):
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            ev = float(expected)
            av = float(actual)
        except Exception as exc:
            pytest.fail(
                f"{ctx}: could not coerce field '{field}' to float "
                f"(expected={expected!r}, actual={actual!r}): {exc}"
            )
        assert math.isclose(av, ev, rel_tol=1e-3, abs_tol=1e-3), (
            f"{ctx}: field '{field}' float mismatch: expected ~{ev}, got {av}."
        )
    else:
        assert actual == expected, (
            f"{ctx}: field '{field}' mismatch: expected {expected!r}, got {actual!r}."
        )


def _check_modified(actual_mod, expected_mod, label):
    assert isinstance(actual_mod, list), (
        f"Expected diff_versions(...)['modified'] to be a list for {label}, "
        f"got {type(actual_mod).__name__}."
    )
    actual_sorted = sorted(actual_mod, key=lambda r: int(r["id"]))
    expected_sorted = sorted(expected_mod, key=lambda r: int(r["id"]))
    assert len(actual_sorted) == len(expected_sorted), (
        f"{label}: 'modified' length mismatch: expected {len(expected_sorted)}, "
        f"got {len(actual_sorted)}; expected ids="
        f"{[int(r['id']) for r in expected_sorted]!r}, actual ids="
        f"{[int(r['id']) for r in actual_sorted]!r}."
    )
    for actual_row, expected_row in zip(actual_sorted, expected_sorted):
        a_id = int(actual_row["id"])
        e_id = int(expected_row["id"])
        assert a_id == e_id, (
            f"{label}: 'modified' id mismatch at sorted position: expected {e_id}, got {a_id}."
        )
        for side in ("old", "new"):
            assert side in actual_row, (
                f"{label}: modified row id={a_id} missing required key '{side}'."
            )
            assert isinstance(actual_row[side], dict), (
                f"{label}: modified row id={a_id}, '{side}' must be a dict, got "
                f"{type(actual_row[side]).__name__}."
            )
            for field, expected_value in expected_row[side].items():
                assert field in actual_row[side], (
                    f"{label}: modified row id={a_id}, '{side}' dict is missing field '{field}'."
                )
                _check_value(
                    field,
                    expected_value,
                    actual_row[side][field],
                    ctx=f"{label} modified row id={a_id} side={side}",
                )


def _assert_diff_result(actual, expected, label):
    assert isinstance(actual, dict), (
        f"{label}: diff_versions must return a dict, got {type(actual).__name__}."
    )
    for key in ("added", "removed", "modified"):
        assert key in actual, f"{label}: missing top-level key '{key}' in returned dict."
    _check_added_removed(actual["added"], expected["added"], f"{label}.added")
    _check_added_removed(actual["removed"], expected["removed"], f"{label}.removed")
    _check_modified(actual["modified"], expected["modified"], label)


def test_diff_v1_to_v4_matches_ground_truth():
    sol = _load_solution_module()
    expected = _load_expected()
    actual = sol.diff_versions(1, 4)
    _assert_diff_result(actual, expected["1_4"], "diff_versions(1, 4)")


def test_diff_v2_to_v3_matches_ground_truth():
    sol = _load_solution_module()
    expected = _load_expected()
    actual = sol.diff_versions(2, 3)
    _assert_diff_result(actual, expected["2_3"], "diff_versions(2, 3)")
    # v2 -> v3 is a delete-only step: there must be no added/modified rows.
    assert sorted(actual["added"]) == [], (
        f"diff_versions(2, 3) expected zero added ids; got {actual['added']!r}."
    )
    assert actual["modified"] == [], (
        f"diff_versions(2, 3) expected zero modified rows (delete-only step); got {actual['modified']!r}."
    )


def test_diff_v3_to_v4_matches_ground_truth():
    sol = _load_solution_module()
    expected = _load_expected()
    actual = sol.diff_versions(3, 4)
    _assert_diff_result(actual, expected["3_4"], "diff_versions(3, 4)")
    # v3 -> v4 is an add-only step: there must be no removed/modified rows.
    assert sorted(actual["removed"]) == [], (
        f"diff_versions(3, 4) expected zero removed ids; got {actual['removed']!r}."
    )
    assert actual["modified"] == [], (
        f"diff_versions(3, 4) expected zero modified rows (add-only step); got {actual['modified']!r}."
    )


def test_diff_is_stable_across_repeated_calls():
    sol = _load_solution_module()
    a = sol.diff_versions(1, 4)
    b = sol.diff_versions(1, 4)

    def _norm(d):
        return {
            "added": sorted(int(x) for x in d.get("added", [])),
            "removed": sorted(int(x) for x in d.get("removed", [])),
            "modified": sorted(
                (
                    int(r["id"]),
                    {k: r["old"].get(k) for k in sorted(r["old"].keys())},
                    {k: r["new"].get(k) for k in sorted(r["new"].keys())},
                )
                for r in d.get("modified", [])
            ),
        }

    assert _norm(a) == _norm(b), (
        "Repeated calls to diff_versions(1, 4) returned different results; "
        "the function must be deterministic and read-only."
    )


def test_table_untouched_after_solution_runs():
    sol = _load_solution_module()
    # Run multiple diffs to exercise checkout(), then make sure the table on
    # disk still has the original 4+ version history and no extra versions
    # have been written by the candidate.
    sol.diff_versions(1, 4)
    sol.diff_versions(2, 3)
    sol.diff_versions(3, 4)
    import lancedb

    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table("customers")
    versions = table.list_versions()
    assert len(versions) >= 4, (
        f"Expected at least 4 versions to remain after running diff_versions; "
        f"got {len(versions)}: {versions!r}."
    )
    # Sanity: the latest checked-out version should still contain at least one row.
    table.checkout_latest()
    assert table.count_rows() > 0, "Latest version of 'customers' is unexpectedly empty."
