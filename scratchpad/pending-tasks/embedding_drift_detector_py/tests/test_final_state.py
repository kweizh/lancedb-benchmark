"""Final-state verification for embedding_drift_detector_py.

The candidate ships:
  * /home/user/myproject/solution.py exposing detect_drift(baseline, current, n_samples=500)
  * /home/user/myproject/run.py — CLI that writes result.json

The verifier independently reimplements the reference algorithm (same seeds,
same epsilon smoothing, same KMeans hyper-parameters) and asserts the
candidate's outputs are numerically equivalent.
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULT_PATH = os.path.join(PROJECT_DIR, "result.json")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
RUN_PATH = os.path.join(PROJECT_DIR, "run.py")

EXPECTED_KEYS = {"kl_divergence", "js_divergence", "drifted", "top_shifted_clusters"}


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set in the verifier environment."
    return rid


def _load_solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"Missing {SOLUTION_PATH}"
    spec = importlib.util.spec_from_file_location("candidate_solution", SOLUTION_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solution"] = mod
    spec.loader.exec_module(mod)
    return mod


def _vectors_from_table(tbl) -> np.ndarray:
    df = tbl.to_pandas()
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in df["vector"].tolist()])
    return arr


def _reference_detect_drift(baseline_vecs: np.ndarray, current_vecs: np.ndarray, n_samples: int = 500) -> dict:
    """Independent reimplementation of the spec, used as ground truth."""
    from sklearn.cluster import KMeans

    rng_b = np.random.default_rng(2026)
    idx_b = rng_b.choice(baseline_vecs.shape[0], size=n_samples, replace=False)
    rng_c = np.random.default_rng(2026)
    idx_c = rng_c.choice(current_vecs.shape[0], size=n_samples, replace=False)

    b_sample = baseline_vecs[idx_b].astype(np.float32, copy=False)
    c_sample = current_vecs[idx_c].astype(np.float32, copy=False)

    km = KMeans(n_clusters=20, random_state=42, n_init=10)
    km.fit(b_sample)
    b_labels = km.predict(b_sample)
    c_labels = km.predict(c_sample)

    p_b = np.bincount(b_labels, minlength=20).astype(np.float64) / n_samples
    p_c = np.bincount(c_labels, minlength=20).astype(np.float64) / n_samples
    p_b = p_b + 1e-12
    p_c = p_c + 1e-12
    p_b = p_b / p_b.sum()
    p_c = p_c / p_c.sum()

    kl = float(np.sum(p_c * np.log(p_c / p_b)))
    m = 0.5 * (p_c + p_b)
    js = float(0.5 * np.sum(p_c * np.log(p_c / m)) + 0.5 * np.sum(p_b * np.log(p_b / m)))

    shifts = np.abs(p_c - p_b)
    order = sorted(range(20), key=lambda c: (-shifts[c], c))
    top5 = order[:5]
    return {
        "kl_divergence": kl,
        "js_divergence": js,
        "drifted": js > 0.05,
        "top_shifted_clusters": list(map(int, top5)),
    }


@pytest.fixture(scope="module")
def candidate_module():
    return _load_solution_module()


@pytest.fixture(scope="module")
def tables():
    import lancedb
    rid = _run_id()
    db = lancedb.connect(DATA_DIR)
    baseline = db.open_table(f"baseline_{rid}")
    current = db.open_table(f"current_{rid}")
    return baseline, current


@pytest.fixture(scope="module")
def reference_result(tables):
    baseline, current = tables
    b_vecs = _vectors_from_table(baseline)
    c_vecs = _vectors_from_table(current)
    assert b_vecs.shape == (1000, 64)
    assert c_vecs.shape == (1000, 64)
    return _reference_detect_drift(b_vecs, c_vecs, n_samples=500)


@pytest.fixture(scope="module")
def candidate_result():
    # Run the candidate-provided driver and load result.json.
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)
    proc = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"`python3 run.py` failed with exit {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert os.path.isfile(RESULT_PATH), "run.py did not create result.json"
    with open(RESULT_PATH) as f:
        return json.load(f)


def test_run_script_exists():
    assert os.path.isfile(RUN_PATH), f"Missing {RUN_PATH}"


def test_solution_exposes_detect_drift(candidate_module):
    fn = getattr(candidate_module, "detect_drift", None)
    assert callable(fn), "solution.py must expose a callable `detect_drift`."


def test_result_json_schema(candidate_result):
    assert isinstance(candidate_result, dict), "result.json must be a JSON object."
    assert set(candidate_result.keys()) == EXPECTED_KEYS, (
        f"result.json keys mismatch. Expected {EXPECTED_KEYS}, got {set(candidate_result.keys())}"
    )


def test_drifted_true(candidate_result):
    assert candidate_result["drifted"] is True, "Expected drifted=True for shifted distribution."


def test_js_divergence_in_bounds(candidate_result):
    js = candidate_result["js_divergence"]
    assert isinstance(js, (int, float)) and not isinstance(js, bool), "js_divergence must be a number."
    assert 0.05 < js < 1.0, f"js_divergence out of expected range: {js}"


def test_kl_divergence_non_negative(candidate_result):
    kl = candidate_result["kl_divergence"]
    assert isinstance(kl, (int, float)) and not isinstance(kl, bool), "kl_divergence must be a number."
    assert kl >= 0.0, f"kl_divergence must be non-negative, got {kl}"


def test_top_shifted_clusters_shape(candidate_result):
    top = candidate_result["top_shifted_clusters"]
    assert isinstance(top, list) and len(top) == 5, "top_shifted_clusters must be a list of 5 ints."
    assert all(isinstance(c, int) and not isinstance(c, bool) for c in top), "All elements must be ints."
    assert len(set(top)) == 5, "Cluster indices must be distinct."
    assert all(0 <= c < 20 for c in top), "Cluster indices must be in [0, 20)."


def test_candidate_matches_reference_top_shifted(candidate_result, reference_result):
    assert candidate_result["top_shifted_clusters"] == reference_result["top_shifted_clusters"], (
        "Candidate's top_shifted_clusters does not match reference (order matters). "
        f"candidate={candidate_result['top_shifted_clusters']} reference={reference_result['top_shifted_clusters']}"
    )


def test_candidate_matches_reference_divergences(candidate_result, reference_result):
    assert abs(candidate_result["js_divergence"] - reference_result["js_divergence"]) < 1e-6, (
        f"js_divergence mismatch: candidate={candidate_result['js_divergence']} "
        f"reference={reference_result['js_divergence']}"
    )
    assert abs(candidate_result["kl_divergence"] - reference_result["kl_divergence"]) < 1e-6, (
        f"kl_divergence mismatch: candidate={candidate_result['kl_divergence']} "
        f"reference={reference_result['kl_divergence']}"
    )


def test_candidate_function_matches_result_json(candidate_module, tables, candidate_result):
    baseline, current = tables
    out = candidate_module.detect_drift(baseline, current, n_samples=500)
    assert isinstance(out, dict), "detect_drift must return a dict."
    assert set(out.keys()) == EXPECTED_KEYS
    assert out["drifted"] == candidate_result["drifted"]
    assert out["top_shifted_clusters"] == candidate_result["top_shifted_clusters"]
    assert abs(out["js_divergence"] - candidate_result["js_divergence"]) < 1e-9
    assert abs(out["kl_divergence"] - candidate_result["kl_divergence"]) < 1e-9


def test_detect_drift_deterministic(candidate_module, tables):
    baseline, current = tables
    a = candidate_module.detect_drift(baseline, current, n_samples=500)
    b = candidate_module.detect_drift(baseline, current, n_samples=500)
    assert a["top_shifted_clusters"] == b["top_shifted_clusters"], "detect_drift must be deterministic."
    assert abs(a["js_divergence"] - b["js_divergence"]) < 1e-12
    assert abs(a["kl_divergence"] - b["kl_divergence"]) < 1e-12
