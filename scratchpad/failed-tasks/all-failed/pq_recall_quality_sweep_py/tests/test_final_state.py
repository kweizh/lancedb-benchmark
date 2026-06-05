import importlib.util
import json
import math
import os
import subprocess
import sys

import lancedb
import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
RUN_SCRIPT_PATH = os.path.join(PROJECT_DIR, "run_sweep.py")
RESULT_PATH = os.path.join(PROJECT_DIR, "result.json")
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")

EXPECTED_KEYS = {4, 8, 16}


def _load_solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"Missing solution module at {SOLUTION_PATH}"
    spec = importlib.util.spec_from_file_location("candidate_solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solution"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep_result():
    # Clean any prior artifact
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)

    # Execute the candidate's CLI to produce result.json.
    proc = subprocess.run(
        ["python3", RUN_SCRIPT_PATH],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, (
        f"run_sweep.py failed with code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )

    module = _load_solution_module()
    assert hasattr(module, "sweep") and callable(module.sweep), (
        "solution.py must expose a callable `sweep()`."
    )
    result = module.sweep()
    return result


def test_sweep_returns_dict(sweep_result):
    assert isinstance(sweep_result, dict), (
        f"sweep() must return a dict, got {type(sweep_result).__name__}."
    )


def test_sweep_keys_exactly_4_8_16(sweep_result):
    keys = {int(k) for k in sweep_result.keys()}
    assert keys == EXPECTED_KEYS, (
        f"sweep() keys must be exactly {EXPECTED_KEYS}, got {keys}."
    )


def test_sweep_values_in_unit_interval(sweep_result):
    for k, v in sweep_result.items():
        assert isinstance(v, float) or isinstance(v, int), (
            f"recall@10 for m={k} must be numeric, got {type(v).__name__}."
        )
        fv = float(v)
        assert math.isfinite(fv), f"recall@10 for m={k} is not finite: {v}."
        assert 0.0 <= fv <= 1.0, (
            f"recall@10 for m={k} must lie in [0.0, 1.0], got {fv}."
        )


def test_sweep_monotonic_in_num_sub_vectors(sweep_result):
    r = {int(k): float(v) for k, v in sweep_result.items()}
    assert r[4] <= r[8] + 1e-9, (
        f"recall must be non-decreasing: r[4]={r[4]} > r[8]={r[8]}."
    )
    assert r[8] <= r[16] + 1e-9, (
        f"recall must be non-decreasing: r[8]={r[8]} > r[16]={r[16]}."
    )
    assert r[16] > r[4], (
        f"recall[16]={r[16]} must be strictly greater than recall[4]={r[4]} "
        "(finer PQ should improve recall on this fixture)."
    )


def test_result_json_matches_sweep(sweep_result):
    assert os.path.isfile(RESULT_PATH), f"Missing artifact {RESULT_PATH}."
    with open(RESULT_PATH) as f:
        on_disk = json.load(f)
    assert isinstance(on_disk, dict), "result.json must be a JSON object."
    disk_keys = {str(k) for k in on_disk.keys()}
    assert disk_keys == {"4", "8", "16"}, (
        f"result.json keys must be exactly {{'4','8','16'}}, got {disk_keys}."
    )
    for k, v in sweep_result.items():
        sk = str(int(k))
        assert sk in on_disk, f"result.json missing key '{sk}'."
        assert abs(float(on_disk[sk]) - float(v)) <= 1e-6, (
            f"result.json[{sk}]={on_disk[sk]} disagrees with in-process sweep()={v}."
        )


def test_lancedb_tables_use_run_id():
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    assert run_id, "ZEALT_RUN_ID must be set in the verifier environment."
    assert os.path.isdir(LANCEDB_DIR), (
        f"Expected LanceDB directory at {LANCEDB_DIR}."
    )
    db = lancedb.connect(LANCEDB_DIR)
    names = db.table_names()
    matched = [n for n in names if run_id in n]
    assert matched, (
        f"Expected at least one LanceDB table whose name includes ZEALT_RUN_ID='{run_id}', "
        f"found tables: {names}."
    )
