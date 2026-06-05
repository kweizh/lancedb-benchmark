import os
import shutil
import subprocess
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_FILE = os.path.join(PROJECT_DIR, "solution.py")
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb")


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid and rid.strip(), "ZEALT_RUN_ID must be set for verification."
    return rid.strip()


def _table_name() -> str:
    return f"bench_{_run_id()}"


@pytest.fixture(scope="session", autouse=True)
def build_solution():
    """Rebuild the LanceDB table + IVF_PQ index from scratch by running the candidate's
    solution.py once at the start of the test session."""
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
        timeout=600,
    )
    assert proc.returncode == 0, (
        "Running `python3 solution.py` failed.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    yield


def _open_table():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    name = _table_name()
    names = db.table_names()
    assert name in names, (
        f"Expected table '{name}' in LanceDB at {LANCEDB_DIR}; found tables: {names}."
    )
    return db.open_table(name)


def test_schema_and_row_count():
    import pyarrow as pa

    tbl = _open_table()
    schema = tbl.schema
    field_names = [f.name for f in schema]
    assert "id" in field_names, f"Missing 'id' column in schema; got fields: {field_names}."
    assert "vector" in field_names, (
        f"Missing 'vector' column in schema; got fields: {field_names}."
    )

    id_field = schema.field("id")
    assert pa.types.is_integer(id_field.type), (
        f"Expected 'id' column to be an integer type, got {id_field.type}."
    )

    vec_field = schema.field("vector")
    vec_type = vec_field.type
    assert pa.types.is_fixed_size_list(vec_type), (
        f"Expected 'vector' to be a fixed_size_list, got {vec_type}."
    )
    assert vec_type.list_size == 384, (
        f"Expected fixed_size_list of length 384 for 'vector', got {vec_type.list_size}."
    )
    assert pa.types.is_floating(vec_type.value_type) and vec_type.value_type.bit_width == 32, (
        f"Expected 'vector' value type to be float32, got {vec_type.value_type}."
    )

    row_count = tbl.count_rows()
    assert row_count >= 1024, (
        f"Expected at least 1024 rows in '{_table_name()}', got {row_count}."
    )


def test_ivf_pq_index_registered():
    tbl = _open_table()
    indices = list(tbl.list_indices())
    assert len(indices) >= 1, (
        f"No indices reported by table.list_indices() for '{_table_name()}'."
    )
    matched = []
    for idx in indices:
        # IndexConfig has .columns (list[str]) and .index_type (str)
        cols = getattr(idx, "columns", None) or []
        itype = getattr(idx, "index_type", "") or ""
        if "vector" in cols and "ivf_pq" in str(itype).lower().replace("-", "_"):
            matched.append(idx)
    assert matched, (
        "Expected at least one IVF_PQ index on the 'vector' column. "
        f"list_indices() returned: {[ (getattr(i, 'name', None), getattr(i, 'columns', None), getattr(i, 'index_type', None)) for i in indices ]}"
    )


def test_evaluate_recall_returns_float_in_range_and_meets_threshold():
    sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    import solution  # type: ignore

    assert hasattr(solution, "evaluate_recall"), (
        "solution module must expose a top-level function `evaluate_recall`."
    )

    recall = solution.evaluate_recall(num_queries=50, k=10)
    assert isinstance(recall, float), (
        f"evaluate_recall must return float, got {type(recall).__name__} (value={recall!r})."
    )
    assert 0.0 <= recall <= 1.0, (
        f"evaluate_recall must return a value in [0.0, 1.0], got {recall}."
    )
    assert recall >= 0.70, (
        f"Recall@10 must be >= 0.70 against brute-force ground truth, got {recall:.4f}."
    )


def test_evaluate_recall_stable_on_repeat_call():
    sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    import solution  # type: ignore

    recall = solution.evaluate_recall(num_queries=20, k=10)
    assert isinstance(recall, float), "evaluate_recall must return a float on repeat calls."
    assert recall >= 0.70, (
        f"Recall@10 must remain >= 0.70 on a second invocation, got {recall:.4f}."
    )
