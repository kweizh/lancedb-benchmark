import importlib
import json
import os
import re
import subprocess
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
GROUND_TRUTH_PATH = os.path.join(PROJECT_DIR, ".ground_truth_outliers.json")


def _run_id() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID env var must be set for verification."
    return run_id


def _table_name() -> str:
    return f"events_{_run_id()}"


@pytest.fixture(scope="session")
def ground_truth_outlier_ids():
    with open(GROUND_TRUTH_PATH) as f:
        ids = json.load(f)
    assert isinstance(ids, list) and len(ids) == 50, (
        f"Ground-truth file at {GROUND_TRUTH_PATH} must contain 50 ids; got {ids!r}."
    )
    return set(int(x) for x in ids)


@pytest.fixture(scope="session")
def run_pipeline_stdout():
    """Execute `python3 run.py` once and return its stdout for downstream tests."""
    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"`python3 run.py` failed with exit code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


def test_stdout_contains_top20_line(run_pipeline_stdout):
    match = re.search(r"^TOP20=(\[.*\])\s*$", run_pipeline_stdout, flags=re.MULTILINE)
    assert match, (
        "Stdout from `python3 run.py` must contain a line starting with `TOP20=` "
        f"followed by a JSON list. Got:\n{run_pipeline_stdout}"
    )
    parsed = json.loads(match.group(1))
    assert isinstance(parsed, list), "TOP20= must be followed by a JSON list."
    assert len(parsed) == 20, f"TOP20 list must contain exactly 20 ids; got {len(parsed)}."
    assert all(isinstance(x, int) for x in parsed), "TOP20 ids must all be integers."


def test_is_outlier_column_exists_and_is_bool(run_pipeline_stdout):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_table_name())
    schema = tbl.schema
    assert "is_outlier" in schema.names, (
        f"`is_outlier` column missing from table; fields are {schema.names!r}."
    )
    field = schema.field("is_outlier")
    assert field.type == pa.bool_(), (
        f"`is_outlier` column must have Arrow type bool; got {field.type!r}."
    )


def test_flagged_count_in_band(run_pipeline_stdout):
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_table_name())
    flagged = tbl.count_rows(filter="is_outlier IS TRUE")
    assert 40 <= flagged <= 60, (
        f"Expected the number of rows flagged as outliers to be in [40, 60]; got {flagged}."
    )


def test_precision_at_50(run_pipeline_stdout, ground_truth_outlier_ids):
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_table_name())
    df = tbl.to_pandas()
    flagged_ids = set(int(x) for x in df.loc[df["is_outlier"] == True, "id"].tolist())  # noqa: E712
    assert len(flagged_ids) > 0, "No rows were flagged as outliers; precision is undefined."
    intersection = flagged_ids & ground_truth_outlier_ids
    precision = len(intersection) / len(flagged_ids)
    assert precision >= 0.90, (
        f"Precision-at-flagged must be >= 0.90; got {precision:.3f} "
        f"(flagged={len(flagged_ids)}, correct={len(intersection)})."
    )


def test_solution_top_outliers_subset(run_pipeline_stdout, ground_truth_outlier_ids):
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        importlib.reload(sys.modules["solution"])
    import solution  # type: ignore

    got = solution.top_outliers(20)
    assert isinstance(got, list), f"top_outliers must return a list; got {type(got)!r}."
    assert len(got) == 20, f"top_outliers(20) must return 20 ids; got {len(got)}."
    assert all(isinstance(x, int) for x in got), "All ids returned by top_outliers must be int."
    missing = set(got) - ground_truth_outlier_ids
    assert not missing, (
        f"top_outliers(20) returned ids that are not in the ground-truth outlier set: {sorted(missing)}."
    )


def test_printed_top20_matches_solution(run_pipeline_stdout):
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        importlib.reload(sys.modules["solution"])
    import solution  # type: ignore

    match = re.search(r"^TOP20=(\[.*\])\s*$", run_pipeline_stdout, flags=re.MULTILINE)
    assert match, "TOP20= line missing in stdout (already checked elsewhere)."
    printed = set(int(x) for x in json.loads(match.group(1)))
    got = set(int(x) for x in solution.top_outliers(20))
    assert printed == got, (
        f"`TOP20=` printed by run.py must match `solution.top_outliers(20)`; "
        f"printed={sorted(printed)} solution={sorted(got)}."
    )


def test_rerun_idempotent(run_pipeline_stdout):
    """`run.py` must not crash on a second invocation (column already exists)."""
    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Second invocation of `python3 run.py` failed with exit {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
