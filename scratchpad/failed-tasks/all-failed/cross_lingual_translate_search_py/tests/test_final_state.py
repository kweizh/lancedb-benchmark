import importlib.util
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")

ZH_QUERY = "什么是机器学习算法"
JA_QUERY = "寿司のレシピを教えてください"
AR_QUERY = "ما هي الصحراء الكبرى"

EXPECTED_TOPICS = {
    "zh": "machine_learning",
    "ja": "sushi_recipes",
    "ar": "sahara_desert",
}


@pytest.fixture(scope="session")
def solution_module():
    assert os.path.isfile(SOLUTION_PATH), f"Candidate solution not found at {SOLUTION_PATH}"
    sys.path.insert(0, PROJECT_DIR)
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    assert spec is not None and spec.loader is not None, "Failed to build import spec for solution.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_solution_exposes_callables(solution_module):
    assert hasattr(solution_module, "translate_to_english"), \
        "solution.py must expose translate_to_english"
    assert callable(solution_module.translate_to_english), \
        "translate_to_english must be callable"
    assert hasattr(solution_module, "cross_lingual_search"), \
        "solution.py must expose cross_lingual_search"
    assert callable(solution_module.cross_lingual_search), \
        "cross_lingual_search must be callable"


def test_translate_to_english_returns_english(solution_module):
    out = solution_module.translate_to_english(ZH_QUERY, "zh")
    assert isinstance(out, str) and out.strip(), \
        f"translate_to_english must return non-empty string; got {out!r}"
    assert any(c.isascii() and c.isalpha() for c in out), \
        f"Translated output should contain ASCII letters; got {out!r}"
    assert "machine learning" in out.lower(), \
        f"Chinese ML query must translate to text containing 'machine learning'; got {out!r}"


def test_verifier_independent_translation():
    from openai import OpenAI

    client = OpenAI()
    chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a literal translator. Translate the user's text to English only. Output the English translation only, with no quotes, prefixes, or commentary.",
            },
            {"role": "user", "content": ZH_QUERY},
        ],
    )
    text = resp.choices[0].message.content.strip()
    assert text, "Verifier OpenAI call must return a non-empty translation."
    assert "machine learning" in text.lower(), \
        f"Verifier OpenAI translation should mention 'machine learning'; got {text!r}"


def _check_search_result(result, expected_topic, k=5):
    assert isinstance(result, list), f"cross_lingual_search must return a list; got {type(result)}"
    assert 1 <= len(result) <= k, f"cross_lingual_search must return between 1 and {k} items; got {len(result)}"
    for i, row in enumerate(result):
        assert isinstance(row, dict), f"Row {i} must be a dict; got {type(row)}"
        for key in ("id", "topic", "content"):
            assert key in row, f"Row {i} missing required key '{key}'; got keys {list(row.keys())}"
    assert result[0]["topic"] == expected_topic, \
        f"Expected rank-1 topic '{expected_topic}'; got '{result[0]['topic']}' (full row: {result[0]})"


def test_cross_lingual_search_zh_machine_learning(solution_module):
    result = solution_module.cross_lingual_search(ZH_QUERY, "zh", k=5)
    _check_search_result(result, EXPECTED_TOPICS["zh"])


def test_cross_lingual_search_ja_sushi(solution_module):
    result = solution_module.cross_lingual_search(JA_QUERY, "ja", k=5)
    _check_search_result(result, EXPECTED_TOPICS["ja"])


def test_cross_lingual_search_ar_sahara(solution_module):
    result = solution_module.cross_lingual_search(AR_QUERY, "ar", k=5)
    _check_search_result(result, EXPECTED_TOPICS["ar"])


def test_verifier_independent_embedding_search():
    import lancedb
    from openai import OpenAI

    uri = os.environ["LANCEDB_URI"]
    prefix = os.environ["LANCEDB_TABLE_PREFIX"]
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"{prefix}{run_id}"

    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)

    client = OpenAI()
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    emb_resp = client.embeddings.create(model=embed_model, input="machine learning algorithms")
    qvec = emb_resp.data[0].embedding
    assert len(qvec) == 1536, f"text-embedding-3-small must produce 1536-d vectors; got {len(qvec)}"

    rows = tbl.search(qvec).limit(5).to_list()
    assert rows, "Direct verifier search returned no rows."
    assert rows[0]["topic"] == "machine_learning", \
        f"Verifier direct embedding+search must yield rank-1 topic 'machine_learning'; got '{rows[0]['topic']}'"
