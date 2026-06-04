import json
import os
import subprocess

import pytest

PROJECT_DIR = "/home/user/cohere_rerank"
RIGGED_ID = "install-linux-guide"


def _run_cli(query: str, language: str, k: int) -> list:
    result = subprocess.run(
        [
            "python3",
            "run_search.py",
            "--query",
            query,
            "--language",
            language,
            "-k",
            str(k),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"run_search.py exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"run_search.py stdout was not valid JSON: {exc}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert isinstance(payload, list), (
        f"run_search.py stdout must be a JSON array, got {type(payload).__name__}: {payload!r}"
    )
    return payload


def _assert_schema(rows, language):
    for i, row in enumerate(rows):
        assert isinstance(row, dict), f"Row {i} is not a JSON object: {row!r}"
        expected_keys = {"id", "content", "language", "rerank_score"}
        actual_keys = set(row.keys())
        assert actual_keys == expected_keys, (
            f"Row {i} must have exactly keys {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
        assert isinstance(row["id"], str), f"Row {i}.id must be a string, got {type(row['id']).__name__}"
        assert isinstance(row["content"], str), (
            f"Row {i}.content must be a string, got {type(row['content']).__name__}"
        )
        assert isinstance(row["language"], str), (
            f"Row {i}.language must be a string, got {type(row['language']).__name__}"
        )
        assert isinstance(row["rerank_score"], (int, float)) and not isinstance(row["rerank_score"], bool), (
            f"Row {i}.rerank_score must be a float, got {type(row['rerank_score']).__name__}"
        )
        assert row["language"] == language, (
            f"Row {i} has language='{row['language']}', expected '{language}' (server-side filter must be applied)."
        )


def _assert_descending(rows):
    scores = [float(r["rerank_score"]) for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"rerank_score is not descending at index {i}: {scores[i]} < {scores[i + 1]}. Full scores: {scores}"
        )


def test_rigged_english_query_returns_ground_truth_rank_1():
    rows = _run_cli("how to install on linux", "en", 5)
    assert len(rows) == 5, f"Expected 5 rows for k=5, got {len(rows)}: {rows!r}"
    _assert_schema(rows, "en")
    _assert_descending(rows)
    assert rows[0]["id"] == RIGGED_ID, (
        f"Expected rigged ground-truth document id='{RIGGED_ID}' at rank 1, "
        f"got id='{rows[0]['id']}'. Full ranking: {[r['id'] for r in rows]}"
    )
    assert float(rows[0]["rerank_score"]) > float(rows[1]["rerank_score"]), (
        "rank-1 rerank_score must be strictly greater than rank-2 rerank_score "
        f"(got {rows[0]['rerank_score']} vs {rows[1]['rerank_score']})."
    )


def test_language_filter_enforces_spanish():
    rows = _run_cli("how to install on linux", "es", 5)
    assert len(rows) > 0, "Expected at least one Spanish result."
    _assert_schema(rows, "es")
    _assert_descending(rows)
    for row in rows:
        assert row["language"] == "es", (
            f"Found row with language='{row['language']}' in es-filtered query — server-side filter failed."
        )


def test_k_truncation_returns_exactly_k_results():
    rows = _run_cli("how to install on linux", "en", 3)
    assert len(rows) == 3, f"Expected exactly 3 rows for k=3, got {len(rows)}: {[r['id'] for r in rows]}"
    _assert_schema(rows, "en")
    _assert_descending(rows)


def test_rerank_scores_are_not_constant():
    rows = _run_cli("how to install on linux", "en", 5)
    scores = [float(r["rerank_score"]) for r in rows]
    assert len(set(round(s, 6) for s in scores)) > 1, (
        f"All rerank_score values are identical ({scores}); real Cohere rerank should "
        f"produce distinct per-document relevance scores."
    )
