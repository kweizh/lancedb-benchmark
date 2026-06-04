import json
import os
import subprocess
import sys

import pytest

PROJECT_DIR = "/home/user/project"
SCRIPT_PATH = os.path.join(PROJECT_DIR, "recommend.py")
QUERY_PATH = os.path.join(PROJECT_DIR, "query.npy")
EXPECTED_PATH = "/opt/zealt/expected.json"
OUT_ALPHA0 = os.path.join(PROJECT_DIR, "out_alpha0.json")
OUT_ALPHA1 = os.path.join(PROJECT_DIR, "out_alpha1.json")


def _load_expected():
    with open(EXPECTED_PATH) as f:
        return json.load(f)


def _run_recommend(alpha: float, output_path: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["BLEND_ALPHA"] = str(alpha)
    if os.path.exists(output_path):
        os.remove(output_path)
    return subprocess.run(
        [
            sys.executable,
            SCRIPT_PATH,
            "--user-id",
            "u_test",
            "--query-vec",
            QUERY_PATH,
            "--k",
            "5",
            "--output",
            output_path,
        ],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_recommend_script_exists():
    assert os.path.isfile(SCRIPT_PATH), (
        f"Candidate must create recommend.py at {SCRIPT_PATH}"
    )


def test_pure_query_ranking_matches_expected():
    """BLEND_ALPHA=0.0 must rank items purely by the supplied query vector."""
    expected = _load_expected()
    result = _run_recommend(0.0, OUT_ALPHA0)
    assert result.returncode == 0, (
        f"recommend.py failed with BLEND_ALPHA=0.0\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert os.path.isfile(OUT_ALPHA0), (
        f"Expected output file {OUT_ALPHA0} was not created."
    )
    with open(OUT_ALPHA0) as f:
        got = json.load(f)
    assert isinstance(got, list) and all(isinstance(x, int) for x in got), (
        f"Output must be a JSON array of integers, got: {got!r}"
    )
    assert len(got) == 5, f"Expected k=5 results, got {len(got)}: {got!r}"
    assert got == expected["expected_alpha0"], (
        f"Pure-query ranking mismatch.\nExpected: {expected['expected_alpha0']}\nGot: {got}"
    )


def test_pure_taste_ranking_matches_expected():
    """BLEND_ALPHA=1.0 must rank items purely by the user's mean-of-history vector."""
    expected = _load_expected()
    result = _run_recommend(1.0, OUT_ALPHA1)
    assert result.returncode == 0, (
        f"recommend.py failed with BLEND_ALPHA=1.0\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert os.path.isfile(OUT_ALPHA1), (
        f"Expected output file {OUT_ALPHA1} was not created."
    )
    with open(OUT_ALPHA1) as f:
        got = json.load(f)
    assert isinstance(got, list) and all(isinstance(x, int) for x in got), (
        f"Output must be a JSON array of integers, got: {got!r}"
    )
    assert len(got) == 5, f"Expected k=5 results, got {len(got)}: {got!r}"
    assert got == expected["expected_alpha1"], (
        f"Pure-taste ranking mismatch.\nExpected: {expected['expected_alpha1']}\nGot: {got}"
    )


def test_blend_alpha_changes_ranking():
    """Sanity check that the two runs produced different orderings,
    proving BLEND_ALPHA actually influenced the result."""
    with open(OUT_ALPHA0) as f:
        a0 = json.load(f)
    with open(OUT_ALPHA1) as f:
        a1 = json.load(f)
    assert a0 != a1, (
        "Pure-query and pure-taste rankings are identical — the candidate likely "
        "ignored the BLEND_ALPHA environment variable."
    )


def test_no_seen_items_returned():
    expected = _load_expected()
    seen = set(expected["seen_item_ids"])
    for out_path in (OUT_ALPHA0, OUT_ALPHA1):
        with open(out_path) as f:
            ids = json.load(f)
        leaked = set(ids) & seen
        assert not leaked, (
            f"{out_path} contains seen item IDs that should have been excluded: {leaked}"
        )
