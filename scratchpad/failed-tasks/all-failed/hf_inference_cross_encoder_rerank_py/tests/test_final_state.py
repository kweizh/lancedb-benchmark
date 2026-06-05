"""Final-state verification for hf_inference_cross_encoder_rerank_py.

Runs after the candidate finishes. Imports `solution` from /home/user/myproject
and exercises `rerank_search` against the seeded 200-row LanceDB `docs` table.
The fixture is rigged so that pure vector search puts `rigged-distractor` above
`rigged-correct` in the top-30, which means a candidate that omits the
cross-encoder rerank step cannot return `rigged-correct` at rank 1.
"""

import json
import os
import subprocess
import sys

import httpx
import lancedb
import numpy as np
import pytest
from openai import OpenAI

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
EXPECTED_JSON = os.path.join(PROJECT_DIR, ".expected.json")
TABLE_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"
HF_RERANKERS = (
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


def _load_expected():
    with open(EXPECTED_JSON) as f:
        return json.load(f)


def _import_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import importlib

    if "solution" in sys.modules:
        importlib.reload(sys.modules["solution"])
    import solution  # type: ignore

    return solution


def test_solution_module_exposes_rerank_search():
    solution = _import_solution()
    fn = getattr(solution, "rerank_search", None)
    assert callable(fn), "solution.py must expose a callable `rerank_search(query, k=10)`."


def test_anchor_query_top1_is_rigged_correct():
    """Core sentinel check: candidate's rerank must put `rigged-correct` at rank 1.

    The image was seeded so that pure vector retrieval puts `rigged-distractor`
    above `rigged-correct` in the top-30. The only way to recover `rigged-correct`
    as rank-1 is to call the Hugging Face cross-encoder reranker.
    """
    expected = _load_expected()
    anchor_query = expected["anchor_query"]
    correct_id = expected["rigged_correct_id"]
    distractor_id = expected["rigged_distractor_id"]

    solution = _import_solution()
    results = solution.rerank_search(anchor_query, k=10)

    assert isinstance(results, list), f"rerank_search must return a list; got {type(results).__name__}."
    assert len(results) == 10, f"rerank_search(k=10) must return exactly 10 items; got {len(results)}."

    allowed_keys = {"id", "content", "rerank_score"}
    for i, row in enumerate(results):
        assert isinstance(row, dict), f"Result element {i} must be a dict; got {type(row).__name__}."
        assert set(row.keys()) == allowed_keys, (
            f"Result element {i} must have keys exactly {allowed_keys!r}; got {set(row.keys())!r}."
        )
        assert isinstance(row["id"], str), f"Element {i} has non-string id."
        assert isinstance(row["content"], str), f"Element {i} has non-string content."
        assert isinstance(row["rerank_score"], (int, float)), (
            f"Element {i} has non-numeric rerank_score: {row['rerank_score']!r}."
        )

    scores = [float(r["rerank_score"]) for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Results not sorted by descending rerank_score; pos {i}={scores[i]} < pos {i+1}={scores[i+1]}."
        )

    ids = [r["id"] for r in results]
    assert ids[0] == correct_id, (
        f"Expected rank-1 id={correct_id!r}; got {ids[0]!r}. Full top-10 ids: {ids!r}. "
        "If the cross-encoder rerank step is skipped, the keyword-stuffed distractor wins; "
        "calling the Hugging Face cross-encoder reranker is required to recover the right answer."
    )
    assert ids[0] != distractor_id, (
        f"Rank-1 is the keyword-stuffed distractor {distractor_id!r}; cross-encoder rerank step appears to be missing."
    )


def test_vector_only_baseline_distractor_outranks_correct():
    """Independent sanity check: prove the fixture is actually rigged.

    The verifier independently embeds the anchor query with OpenAI
    `text-embedding-3-small` and runs a plain vector search top-30 on the
    `docs` table. Both rigged docs must be in the top-30, and `rigged-distractor`
    must rank strictly above `rigged-correct`.
    """
    expected = _load_expected()
    anchor_query = expected["anchor_query"]
    correct_id = expected["rigged_correct_id"]
    distractor_id = expected["rigged_distractor_id"]

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set in the verifier environment."

    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[anchor_query])
    qvec = np.asarray(resp.data[0].embedding, dtype=np.float32)

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    rows = tbl.search(qvec).limit(30).to_list()
    ids = [r["id"] for r in rows]

    assert correct_id in ids, f"Vector-only top-30 missing {correct_id!r}; got {ids!r}."
    assert distractor_id in ids, f"Vector-only top-30 missing {distractor_id!r}; got {ids!r}."

    rank_correct = ids.index(correct_id)
    rank_distractor = ids.index(distractor_id)
    assert rank_distractor < rank_correct, (
        "Fixture invariant violated: vector-only ranking is expected to put the keyword-stuffed "
        f"distractor strictly above the correct doc. Got rank(distractor)={rank_distractor}, "
        f"rank(correct)={rank_correct}. Vector-only top-5 ids: {ids[:5]!r}."
    )


def test_k_parameter_is_honored():
    expected = _load_expected()
    anchor_query = expected["anchor_query"]
    correct_id = expected["rigged_correct_id"]

    solution = _import_solution()
    results = solution.rerank_search(anchor_query, k=5)
    assert isinstance(results, list), "rerank_search must return a list."
    assert len(results) == 5, f"rerank_search(k=5) must return exactly 5 items; got {len(results)}."
    assert results[0]["id"] == correct_id, (
        f"Expected rank-1 id={correct_id!r} for k=5; got {results[0]['id']!r}."
    )


def test_hf_inference_api_reachable_from_verifier():
    """Verifier-side credential / network sanity check on the Hugging Face Inference API.

    Tries each candidate reranker until one returns a parseable score payload.
    This guarantees that, at evaluation time, the HF endpoint and the HF_TOKEN
    that the candidate sees are actually working.
    """
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN must be set in the verifier environment."

    pair_payload = {
        "inputs": {"source_sentence": "ping", "sentences": ["pong"]},
        "options": {"wait_for_model": True},
    }
    text_pair_payload = {
        "inputs": [["ping", "pong"]],
        "options": {"wait_for_model": True},
    }

    last_err = None
    ok = False
    with httpx.Client(timeout=60.0) as client:
        for model in HF_RERANKERS:
            url = f"https://api-inference.huggingface.co/models/{model}"
            for payload in (pair_payload, text_pair_payload):
                try:
                    r = client.post(
                        url,
                        headers={"Authorization": f"Bearer {hf_token}"},
                        json=payload,
                    )
                    if r.status_code == 200:
                        ok = True
                        break
                    last_err = f"{model} payload-shape -> HTTP {r.status_code}: {r.text[:300]}"
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{model}: {exc!r}"
            if ok:
                break

    assert ok, (
        f"Hugging Face Inference API did not return HTTP 200 for any candidate reranker. "
        f"Last error: {last_err}"
    )


def test_rerank_search_requires_hf_token_at_runtime():
    """Re-run rerank_search in a fresh subprocess with HF_TOKEN unset.

    The pipeline must fail; that is the proof that the candidate is genuinely
    calling the Hugging Face Inference API (not silently using a cached or
    offline reranker).
    """
    expected = _load_expected()
    anchor_query = expected["anchor_query"]

    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.pop("HUGGINGFACE_API_KEY", None)
    env.pop("HUGGINGFACEHUB_API_TOKEN", None)

    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {PROJECT_DIR!r})\n"
        "import solution\n"
        f"r = solution.rerank_search({anchor_query!r}, k=5)\n"
        "print(json.dumps([x['id'] for x in r]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, (
        "rerank_search succeeded with HF_TOKEN unset; the candidate appears to skip the Hugging Face "
        f"rerank step entirely. stdout={result.stdout!r}, stderr={result.stderr[-500:]!r}."
    )
