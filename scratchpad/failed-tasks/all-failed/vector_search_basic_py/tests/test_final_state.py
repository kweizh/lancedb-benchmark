import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest


PROJECT_DIR = "/workspace/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
RESULTS_PATH = "/workspace/output/results.json"
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")


def _expected_top5():
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((12, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    dists = np.linalg.norm(vectors - query, axis=1)
    order = np.argsort(dists, kind="stable")[:5]
    return [(int(i), float(dists[i])) for i in order]


@pytest.fixture(scope="session")
def run_solution():
    """Re-run the candidate solution from a clean state, then return its results.json."""
    # Clean previous artifacts so this verification is deterministic.
    subprocess.run(["rm", "-rf", LANCEDB_URI, "/workspace/output"], check=False)
    assert os.path.isfile(SOLUTION_PATH), (
        f"Candidate solution script not found at {SOLUTION_PATH}. The executor must produce it."
    )
    result = subprocess.run(
        [sys.executable, SOLUTION_PATH],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )
    assert result.returncode == 0, (
        f"solution.py exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert os.path.isfile(RESULTS_PATH), (
        f"Expected results file {RESULTS_PATH} to exist after running solution.py."
    )
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def test_results_is_list_of_five(run_solution):
    data = run_solution
    assert isinstance(data, list), f"Expected a JSON list, got {type(data).__name__}."
    assert len(data) == 5, f"Expected 5 results, got {len(data)}."


def test_each_row_has_correct_keys_and_types(run_solution):
    data = run_solution
    required_keys = {"id", "text", "_distance"}
    for idx, row in enumerate(data):
        assert isinstance(row, dict), f"Row {idx} is not a JSON object."
        assert set(row.keys()) == required_keys, (
            f"Row {idx} must contain exactly keys {required_keys}, got {set(row.keys())}."
        )
        assert isinstance(row["id"], int) and not isinstance(row["id"], bool), (
            f"Row {idx} 'id' must be an int, got {type(row['id']).__name__}."
        )
        assert isinstance(row["text"], str), (
            f"Row {idx} 'text' must be a string, got {type(row['text']).__name__}."
        )
        assert isinstance(row["_distance"], float), (
            f"Row {idx} '_distance' must be a float, got {type(row['_distance']).__name__}."
        )


def test_ids_match_expected_topk(run_solution):
    data = run_solution
    expected = _expected_top5()
    expected_ids = [eid for eid, _ in expected]
    actual_ids = [row["id"] for row in data]
    assert actual_ids == expected_ids, (
        f"Top-5 id ordering mismatch. Expected {expected_ids}, got {actual_ids}."
    )


def test_text_matches_id_pattern(run_solution):
    data = run_solution
    for row in data:
        expected_text = f"document_{row['id']}"
        assert row["text"] == expected_text, (
            f"Expected text {expected_text!r} for id={row['id']}, got {row['text']!r}."
        )


def test_distances_are_sorted_and_close(run_solution):
    data = run_solution
    expected = _expected_top5()
    distances = [row["_distance"] for row in data]
    assert distances == sorted(distances), (
        f"_distance values must be non-decreasing, got {distances}."
    )
    for row, (_eid, edist) in zip(data, expected):
        assert abs(row["_distance"] - edist) < 1e-3, (
            f"_distance for id={row['id']} expected ~{edist}, got {row['_distance']}."
        )


def test_lancedb_table_schema_and_rowcount():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    table_names = db.table_names()
    assert "documents" in table_names, (
        f"Expected a 'documents' table in LanceDB at {LANCEDB_URI}, found tables: {table_names}."
    )
    table = db.open_table("documents")
    arrow_tbl = table.to_arrow()

    assert arrow_tbl.num_rows == 12, (
        f"Expected exactly 12 rows in 'documents', got {arrow_tbl.num_rows}."
    )

    schema = arrow_tbl.schema
    assert "id" in schema.names, "Schema is missing the 'id' column."
    assert "text" in schema.names, "Schema is missing the 'text' column."
    assert "vector" in schema.names, "Schema is missing the 'vector' column."

    id_field = schema.field("id")
    assert pa.types.is_int64(id_field.type), (
        f"'id' column must be int64, got {id_field.type}."
    )

    text_field = schema.field("text")
    assert pa.types.is_string(text_field.type) or pa.types.is_large_string(text_field.type), (
        f"'text' column must be string/utf8, got {text_field.type}."
    )

    vector_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type), (
        f"'vector' column must be fixed_size_list, got {vector_field.type}."
    )
    assert vector_field.type.list_size == 8, (
        f"'vector' fixed_size_list length must be 8, got {vector_field.type.list_size}."
    )
    assert pa.types.is_float32(vector_field.type.value_type), (
        f"'vector' inner element type must be float32, got {vector_field.type.value_type}."
    )

    ids = arrow_tbl.column("id").to_pylist()
    assert sorted(ids) == list(range(12)), (
        f"Expected ids to be exactly 0..11, got {sorted(ids)}."
    )
