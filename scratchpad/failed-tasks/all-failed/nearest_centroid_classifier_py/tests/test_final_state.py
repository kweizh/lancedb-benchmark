import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = "/app/lancedb_data"
TEST_SET_PATH = "/app/test_set.json"


@pytest.fixture(scope="session")
def run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set in the verifier environment."
    return rid


@pytest.fixture(scope="session")
def test_set():
    with open(TEST_SET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def solution_module():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    spec_path = os.path.join(PROJECT_DIR, "solution.py")
    assert os.path.isfile(spec_path), f"solution.py must exist at {spec_path}."
    spec = importlib.util.spec_from_file_location("solution", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def run_driver_output():
    # Clean prior log
    log_path = os.path.join(PROJECT_DIR, "accuracy.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    result = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=300,
    )
    return result


def test_run_driver_executes(run_driver_output):
    assert run_driver_output.returncode == 0, (
        f"`python3 run.py` exited with code {run_driver_output.returncode}.\n"
        f"stdout:\n{run_driver_output.stdout}\nstderr:\n{run_driver_output.stderr}"
    )


def test_run_driver_prints_accuracy(run_driver_output):
    out = run_driver_output.stdout
    matches = re.findall(r"accuracy=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", out)
    assert matches, (
        f"`python3 run.py` stdout must contain a line like 'accuracy=<float>'. Got:\n{out}"
    )
    acc = float(matches[-1])
    assert 0.0 <= acc <= 1.0, f"Reported accuracy {acc} is not in [0.0, 1.0]."
    assert acc >= 0.90, f"Reported accuracy {acc} is below the 0.90 threshold."


def test_centroids_table_exists_with_correct_schema(run_driver_output, run_id):
    import lancedb

    assert run_driver_output.returncode == 0, "Driver must succeed before checking centroids."
    db = lancedb.connect(LANCEDB_DIR)
    name = f"centroids_{run_id}"
    assert name in db.table_names(), (
        f"Centroids table '{name}' missing from {LANCEDB_DIR}. "
        f"Existing tables: {db.table_names()}"
    )
    tbl = db.open_table(name)
    assert tbl.count_rows() == 6, (
        f"Centroids table must contain exactly 6 rows; got {tbl.count_rows()}."
    )
    schema = tbl.schema
    label_field = schema.field("label")
    assert str(label_field.type) == "int32", (
        f"Centroids 'label' field must be Int32; got {label_field.type}."
    )
    vector_field = schema.field("vector")
    assert vector_field.type.list_size == 40, (
        f"Centroids 'vector' must be fixed_size_list of size 40; got {vector_field.type}."
    )
    assert "float" in str(vector_field.type.value_type), (
        f"Centroids 'vector' must hold floats; got {vector_field.type.value_type}."
    )

    df = tbl.to_pandas()
    labels = sorted(int(x) for x in df["label"].tolist())
    assert labels == [0, 1, 2, 3, 4, 5], (
        f"Centroids labels must be exactly [0..5]; got {labels}."
    )


def test_centroids_match_independent_recomputation(run_driver_output, run_id):
    import lancedb

    assert run_driver_output.returncode == 0, "Driver must succeed before centroid comparison."
    db = lancedb.connect(LANCEDB_DIR)
    train = db.open_table(f"train_data_{run_id}").to_pandas()
    cent = db.open_table(f"centroids_{run_id}").to_pandas()

    train_vecs = np.array(train["vector"].tolist(), dtype=np.float64)
    train_labels = np.array(train["label"].tolist(), dtype=np.int64)

    expected = {}
    for c in range(6):
        mask = train_labels == c
        assert mask.sum() == 100, f"Training class {c} must have exactly 100 rows."
        expected[c] = train_vecs[mask].mean(axis=0)

    actual = {int(row["label"]): np.array(row["vector"], dtype=np.float64)
              for _, row in cent.iterrows()}
    for c in range(6):
        assert c in actual, f"Centroid for label {c} is missing."
        assert np.allclose(actual[c], expected[c], atol=1e-4), (
            f"Centroid for class {c} differs from independent recomputation.\n"
            f"max abs diff = {np.max(np.abs(actual[c] - expected[c]))}"
        )


def test_classify_returns_valid_int(run_driver_output, solution_module, test_set):
    assert run_driver_output.returncode == 0
    vec = test_set[0]["vector"]
    pred = solution_module.classify(vec)
    assert isinstance(pred, (int, np.integer)), (
        f"classify must return an int; got {type(pred)}."
    )
    assert 0 <= int(pred) <= 5, f"classify must return label in {{0..5}}; got {pred}."


def test_evaluate_accuracy_threshold(run_driver_output, solution_module, test_set):
    assert run_driver_output.returncode == 0
    acc = solution_module.evaluate(test_set)
    assert isinstance(acc, float), f"evaluate must return a float; got {type(acc)}."
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} not in [0,1]."
    assert acc >= 0.90, f"Accuracy {acc} below the 0.90 threshold."


def test_build_centroids_is_idempotent(run_driver_output, solution_module, run_id):
    import lancedb

    assert run_driver_output.returncode == 0
    # Call build a second time.
    solution_module.build_centroids()

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(f"centroids_{run_id}")
    assert tbl.count_rows() == 6, (
        f"After a second build_centroids call the table must still have 6 rows; "
        f"got {tbl.count_rows()}."
    )

    cent = tbl.to_pandas()
    train = db.open_table(f"train_data_{run_id}").to_pandas()
    train_vecs = np.array(train["vector"].tolist(), dtype=np.float64)
    train_labels = np.array(train["label"].tolist(), dtype=np.int64)
    for _, row in cent.iterrows():
        c = int(row["label"])
        expected = train_vecs[train_labels == c].mean(axis=0)
        actual = np.array(row["vector"], dtype=np.float64)
        assert np.allclose(actual, expected, atol=1e-4), (
            f"After idempotent re-run, centroid {c} drifted from independent mean."
        )
