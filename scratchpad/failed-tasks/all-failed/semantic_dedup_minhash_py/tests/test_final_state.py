import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
RESULT_PATH = os.path.join(PROJECT_DIR, "result.json")
LANCEDB_DIR = "/app/lancedb_data"
GROUND_TRUTH_PATH = "/app/ground_truth_pairs.json"


@pytest.fixture(scope="module")
def run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID environment variable must be set."
    return rid


@pytest.fixture(scope="module")
def table_name(run_id):
    return f"documents_{run_id}"


@pytest.fixture(scope="module")
def ground_truth_pairs():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = [tuple(sorted(p)) for p in data["pairs"]]
    return pairs


@pytest.fixture(scope="module")
def run_cli():
    # Remove any stale result.json before running the candidate's CLI.
    if os.path.isfile(RESULT_PATH):
        os.remove(RESULT_PATH)
    proc = subprocess.run(
        ["python3", "run.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env=os.environ.copy(),
    )
    return proc


def test_cli_exit_code_and_stdout(run_cli):
    assert run_cli.returncode == 0, (
        f"`python3 run.py` exited with code {run_cli.returncode}.\n"
        f"stdout:\n{run_cli.stdout}\nstderr:\n{run_cli.stderr}"
    )
    stdout_lines = [ln for ln in run_cli.stdout.strip().splitlines() if ln.strip()]
    assert stdout_lines, f"Expected at least one stdout line, got empty stdout. stderr: {run_cli.stderr}"
    last = stdout_lines[-1].strip()
    assert re.fullmatch(r"num_components=\d+", last), (
        f"Expected last stdout line to match 'num_components=<int>', got: {last!r}."
    )


def test_result_json_exists_and_shape(run_cli):
    assert os.path.isfile(RESULT_PATH), f"Expected result file at {RESULT_PATH} to exist."
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), f"Expected result.json to contain a JSON object, got {type(data).__name__}."
    assert set(data.keys()) == {"num_components", "components"}, (
        f"Expected exactly keys {{'num_components','components'}}, got {set(data.keys())!r}."
    )
    assert isinstance(data["num_components"], int), "num_components must be an int."
    assert isinstance(data["components"], list), "components must be a list."
    for i, comp in enumerate(data["components"]):
        assert isinstance(comp, list), f"components[{i}] must be a list, got {type(comp).__name__}."
        for j, x in enumerate(comp):
            assert isinstance(x, int), f"components[{i}][{j}] must be an int, got {type(x).__name__}."


def test_total_components_is_250(run_cli):
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_components"] == 250, (
        f"Expected num_components=250 for the seeded corpus, got {data['num_components']}."
    )
    assert len(data["components"]) == 250, (
        f"Expected len(components)=250, got {len(data['components'])}."
    )


def test_each_id_appears_exactly_once(run_cli):
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen = []
    for comp in data["components"]:
        seen.extend(comp)
    assert len(seen) == 300, f"Expected 300 ids across components, got {len(seen)}."
    assert sorted(seen) == list(range(300)), (
        "Each id in [0, 300) must appear exactly once across all components."
    )


def test_duplicate_pairs_collapsed(run_cli, ground_truth_pairs):
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    components_as_sets = [frozenset(c) for c in data["components"]]
    for a, b in ground_truth_pairs:
        target = frozenset({a, b})
        assert target in components_as_sets, (
            f"Expected duplicate pair {{{a}, {b}}} to appear as a length-2 component, "
            f"but it was not found among the {len(components_as_sets)} returned components."
        )


def test_singletons_for_non_duplicates(run_cli, ground_truth_pairs):
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    duplicate_ids = set()
    for a, b in ground_truth_pairs:
        duplicate_ids.add(a)
        duplicate_ids.add(b)
    singletons = {c[0] for c in data["components"] if len(c) == 1}
    expected_singletons = set(range(300)) - duplicate_ids
    missing = expected_singletons - singletons
    assert not missing, (
        f"Expected all non-duplicate ids to appear as singletons; missing as singleton: {sorted(missing)[:20]}..."
    )


def test_dedupe_is_deterministic(table_name):
    # Import the candidate's solution module fresh.
    sys.path.insert(0, PROJECT_DIR)
    try:
        if "solution" in sys.modules:
            del sys.modules["solution"]
        solution = importlib.import_module("solution")
    finally:
        if PROJECT_DIR in sys.path:
            sys.path.remove(PROJECT_DIR)

    assert hasattr(solution, "dedupe") and callable(solution.dedupe), (
        "solution.dedupe must be a callable function."
    )

    result_a = solution.dedupe(LANCEDB_DIR, table_name)
    result_b = solution.dedupe(LANCEDB_DIR, table_name)

    assert isinstance(result_a, dict) and isinstance(result_b, dict), (
        "solution.dedupe must return a dict."
    )
    assert result_a.get("num_components") == result_b.get("num_components"), (
        "solution.dedupe must be deterministic: num_components differed across two runs."
    )
    assert result_a.get("components") == result_b.get("components"), (
        "solution.dedupe must be deterministic: components differed across two runs."
    )
    assert result_a["num_components"] == 250, (
        f"solution.dedupe must return num_components=250 on the seeded fixture, got {result_a['num_components']}."
    )


def test_datasketch_minhashlsh_used():
    # Smoke-check that datasketch is installed and importable in the verifier
    # environment so the implementation contract is enforceable.
    ds = importlib.import_module("datasketch")
    assert hasattr(ds, "MinHashLSH"), "datasketch.MinHashLSH must be importable."
    assert hasattr(ds, "MinHash"), "datasketch.MinHash must be importable."


def test_solution_module_imports():
    sys.path.insert(0, PROJECT_DIR)
    try:
        spec = importlib.util.find_spec("solution")
        assert spec is not None, f"Expected solution.py to be importable from {PROJECT_DIR}."
    finally:
        if PROJECT_DIR in sys.path:
            sys.path.remove(PROJECT_DIR)
