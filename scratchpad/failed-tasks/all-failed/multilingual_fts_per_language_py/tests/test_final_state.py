import importlib
import os
import subprocess
import sys

import pytest


PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


def _run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID environment variable is not set."
    return rid


def _table_name(lang: str) -> str:
    return f"docs_{lang}_{_run_id()}"


@pytest.fixture(scope="module")
def solution_module():
    """Import the candidate's solution.py once and expose the module object."""
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Drop any cached copy from a previous test session.
    sys.modules.pop("solution", None)
    mod = importlib.import_module("solution")
    return mod


def test_module_imports_cleanly():
    """`python3 -c \"import solution\"` exits 0 — this triggers FTS index creation."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import solution"],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "Fresh `import solution` failed:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_module_reimport_idempotent():
    """Re-importing/reloading the module must NOT raise 'index already exists'."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import importlib, solution; "
        "importlib.reload(solution); "
        "importlib.reload(solution); "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "Re-import of solution was not idempotent:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "ok" in result.stdout, (
        f"Expected 'ok' marker in stdout, got: {result.stdout!r}"
    )


def test_search_per_lang_is_callable(solution_module):
    assert hasattr(solution_module, "search_per_lang"), (
        "solution.py does not expose a `search_per_lang` symbol."
    )
    assert callable(solution_module.search_per_lang), (
        "solution.search_per_lang is not callable."
    )


def test_fts_index_present_each_table(solution_module):
    """Each of the three tables must report at least one FTS index after import."""
    import lancedb

    db = lancedb.connect(DB_DIR)
    for lang in ("en", "de", "zh"):
        tname = _table_name(lang)
        tbl = db.open_table(tname)
        indices = tbl.list_indices()
        assert len(indices) >= 1, (
            f"Table {tname!r} has no indices after `import solution`; "
            f"expected at least one FTS index."
        )
        # At least one index must look like an FTS index.
        types = [str(getattr(ix, "index_type", "")).upper() for ix in indices]
        assert any("FTS" in t or "INVERTED" in t for t in types), (
            f"Table {tname!r} has no FTS-like index. Reported index_types: {types}"
        )


def test_english_running_shoes_rank1(solution_module):
    """English BM25 puts the rigged running-shoes doc (id=1) at rank-1."""
    out = solution_module.search_per_lang("running shoes", "en", k=5)
    assert isinstance(out, list), f"Expected list, got {type(out).__name__}: {out!r}"
    assert len(out) == 5, f"Expected 5 results, got {len(out)}: {out!r}"
    for v in out:
        assert isinstance(v, int) and not isinstance(v, bool), (
            f"Result list must contain plain ints. Got element {v!r} of "
            f"type {type(v).__name__}. Full list: {out!r}"
        )
    assert out[0] == 1, (
        f"Expected English rigged doc id=1 at rank-1 for query 'running shoes'. "
        f"Got top-5={out!r}."
    )


def test_german_laufen_rank1(solution_module):
    """German BM25 with stemming puts the rigged 'laufen' doc (id=1) at rank-1."""
    out = solution_module.search_per_lang("laufen", "de", k=5)
    assert isinstance(out, list), f"Expected list, got {type(out).__name__}: {out!r}"
    assert len(out) == 5, f"Expected 5 results, got {len(out)}: {out!r}"
    for v in out:
        assert isinstance(v, int) and not isinstance(v, bool), (
            f"Result list must contain plain ints. Got element {v!r} of "
            f"type {type(v).__name__}. Full list: {out!r}"
        )
    assert out[0] == 1, (
        f"Expected German rigged doc id=1 at rank-1 for query 'laufen'. "
        f"Got top-5={out!r}."
    )


def test_chinese_pao_bu_rank1(solution_module):
    """Chinese BM25 via jieba pre-tokenization returns id=1 at rank-1 for '跑步'."""
    out = solution_module.search_per_lang("跑步", "zh", k=5)
    assert isinstance(out, list), f"Expected list, got {type(out).__name__}: {out!r}"
    assert len(out) == 5, f"Expected 5 results, got {len(out)}: {out!r}"
    for v in out:
        assert isinstance(v, int) and not isinstance(v, bool), (
            f"Result list must contain plain ints. Got element {v!r} of "
            f"type {type(v).__name__}. Full list: {out!r}"
        )
    assert out[0] == 1, (
        f"Expected Chinese rigged doc id=1 at rank-1 for query '跑步'. "
        f"Got top-5={out!r}."
    )


def test_unsupported_language_raises(solution_module):
    with pytest.raises(ValueError):
        solution_module.search_per_lang("foo", "jp", k=5)


def test_result_length_respects_k(solution_module):
    """When k <= table size, returned list length must equal k."""
    out_en = solution_module.search_per_lang("the", "en", k=3)
    assert isinstance(out_en, list) and len(out_en) == 3, (
        f"English k=3 must yield a list of length 3, got {out_en!r}"
    )
    out_zh = solution_module.search_per_lang("跑步", "zh", k=2)
    assert isinstance(out_zh, list) and len(out_zh) == 2, (
        f"Chinese k=2 must yield a list of length 2, got {out_zh!r}"
    )
