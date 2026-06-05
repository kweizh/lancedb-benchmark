import importlib.util
import os
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
LANCEDB_DIR = "/app/lancedb_data"

QUERY = "How do vector databases handle production retrieval pipelines?"
EMBED_MODEL = "text-embedding-3-small"


def _load_solution():
    """Load /home/user/myproject/solution.py as a module."""
    assert os.path.isfile(SOLUTION_PATH), f"Candidate solution missing at {SOLUTION_PATH}"
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "retrieve_and_pack"), \
        "solution.py must define retrieve_and_pack(query, max_tokens)."
    return module


def _table_name():
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID environment variable is not set."
    return f"chunks_{run_id}"


def _open_chunks_table():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    return db.open_table(_table_name())


def _embed_query(query: str):
    """Embed the query with the real OpenAI API, matching the seed model."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=query)
    return resp.data[0].embedding


def _candidate_pool(query_vec, top_k: int = 80):
    """Fetch top-k LanceDB candidates with full chunk metadata."""
    tbl = _open_chunks_table()
    rows = (
        tbl.search(query_vec)
        .distance_type("cosine")
        .limit(top_k)
        .to_list()
    )
    return rows


def _reference_pack(rows, max_tokens, use_diversity=True):
    """Reference greedy packer: descending relevance with topic diversity rule."""
    rows_sorted = sorted(rows, key=lambda r: r["_distance"])
    selected = []
    topic_count = {}
    total = 0
    for r in rows_sorted:
        toks = int(r["tokens"])
        if total + toks > max_tokens:
            continue
        if use_diversity and topic_count.get(int(r["topic_id"]), 0) >= 2:
            continue
        selected.append(r)
        total += toks
        topic_count[int(r["topic_id"])] = topic_count.get(int(r["topic_id"]), 0) + 1
    return selected, total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_solution_module_importable():
    _load_solution()


def test_schema_sanity():
    tbl = _open_chunks_table()
    assert tbl.count_rows() == 150, f"Expected 150 chunks, found {tbl.count_rows()}."
    schema_names = {f.name for f in tbl.schema}
    expected = {"id", "text", "topic_id", "tokens", "embedding"}
    missing = expected - schema_names
    assert not missing, f"Seeded table missing expected columns: {missing}."


def test_functional_contract_at_design_budget():
    mod = _load_solution()
    result = mod.retrieve_and_pack(QUERY, 600)

    assert isinstance(result, dict), "retrieve_and_pack must return a dict."
    assert "chunks" in result and "total_tokens" in result, \
        "Result must contain 'chunks' and 'total_tokens' keys."

    chunks = result["chunks"]
    total = result["total_tokens"]

    assert isinstance(chunks, list), "result['chunks'] must be a list."
    assert isinstance(total, int), "result['total_tokens'] must be an int."

    required_keys = {"id", "text", "topic_id", "tokens", "score"}
    for i, c in enumerate(chunks):
        missing = required_keys - set(c.keys())
        assert not missing, f"chunk {i} missing keys {missing}; got {list(c.keys())}"

    assert total == sum(int(c["tokens"]) for c in chunks), \
        f"total_tokens ({total}) must equal sum of chunk tokens."
    assert total <= 600, f"total_tokens {total} exceeds budget 600."

    # No topic appears > 2 times.
    topic_counts = {}
    for c in chunks:
        t = int(c["topic_id"])
        topic_counts[t] = topic_counts.get(t, 0) + 1
    over = {t: n for t, n in topic_counts.items() if n > 2}
    assert not over, f"topic_id appears > 2 times: {over}"

    # Per-chunk tiktoken validation.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        expected_tok = len(enc.encode(c["text"]))
        assert int(c["tokens"]) == expected_tok, (
            f"chunk id={c['id']} tokens={c['tokens']} does not match "
            f"tiktoken cl100k_base count {expected_tok}."
        )

    # Sorted in ascending score (descending relevance).
    scores = [float(c["score"]) for c in chunks]
    assert scores == sorted(scores), \
        f"chunks must be in ascending score (descending relevance) order; got {scores}"


def test_ground_truth_match_and_diversity_triggered():
    mod = _load_solution()
    qvec = _embed_query(QUERY)
    rows = _candidate_pool(qvec, top_k=80)

    ref_sel, ref_total = _reference_pack(rows, 600, use_diversity=True)
    ref_ids = [int(r["id"]) for r in ref_sel]

    # Diversity rule must actually trigger.
    nd_sel, _ = _reference_pack(rows, 600, use_diversity=False)
    nd_ids = [int(r["id"]) for r in nd_sel]
    assert ref_ids != nd_ids, (
        "Diversity-enabled and diversity-disabled packs produce identical "
        f"selections at budget 600: {ref_ids}. The rigged fixture failed to "
        "trigger the diversity rule."
    )

    result = mod.retrieve_and_pack(QUERY, 600)
    cand_ids = [int(c["id"]) for c in result["chunks"]]
    assert cand_ids == ref_ids, (
        f"Candidate ids {cand_ids} do not match reference ids {ref_ids} "
        f"at budget 600 (no-diversity ids = {nd_ids})."
    )
    assert int(result["total_tokens"]) == int(ref_total), (
        f"Candidate total_tokens {result['total_tokens']} != reference {ref_total}."
    )


@pytest.mark.parametrize("budget", [5, 30, 50])
def test_tiny_budget_graceful_degradation(budget):
    mod = _load_solution()
    result = mod.retrieve_and_pack(QUERY, budget)
    chunks = result["chunks"]
    assert len(chunks) <= 1, (
        f"At max_tokens={budget}, expected at most 1 chunk, got {len(chunks)}."
    )
    assert int(result["total_tokens"]) == sum(int(c["tokens"]) for c in chunks), \
        "total_tokens must equal sum of chunk tokens at small budgets."
    assert int(result["total_tokens"]) <= budget, \
        f"At tiny budget {budget}, total_tokens={result['total_tokens']} exceeds budget."


def test_mid_budget_honesty():
    mod = _load_solution()
    budget = 1200
    result = mod.retrieve_and_pack(QUERY, budget)

    chunks = result["chunks"]
    total = int(result["total_tokens"])
    assert total == sum(int(c["tokens"]) for c in chunks), \
        f"total_tokens ({total}) must equal sum of chunk tokens."
    assert total <= budget, f"total_tokens {total} exceeds budget {budget}."

    topic_counts = {}
    for c in chunks:
        t = int(c["topic_id"])
        topic_counts[t] = topic_counts.get(t, 0) + 1
    over = {t: n for t, n in topic_counts.items() if n > 2}
    assert not over, f"topic_id appears > 2 times at budget {budget}: {over}"

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        expected_tok = len(enc.encode(c["text"]))
        assert int(c["tokens"]) == expected_tok, (
            f"chunk id={c['id']} tokens={c['tokens']} mismatches tiktoken count {expected_tok}."
        )

    qvec = _embed_query(QUERY)
    rows = _candidate_pool(qvec, top_k=120)
    ref_sel, _ = _reference_pack(rows, budget, use_diversity=True)
    ref_ids = [int(r["id"]) for r in ref_sel]
    cand_ids = [int(c["id"]) for c in chunks]
    assert cand_ids == ref_ids, (
        f"Candidate ids {cand_ids} do not match reference ids {ref_ids} at budget {budget}."
    )


def test_no_text_truncation_or_rewriting():
    mod = _load_solution()
    result = mod.retrieve_and_pack(QUERY, 600)

    tbl = _open_chunks_table()
    by_id = {int(r["id"]): r["text"] for r in tbl.to_arrow().to_pylist()}

    for c in result["chunks"]:
        cid = int(c["id"])
        assert cid in by_id, f"Returned chunk id={cid} not present in seeded table."
        assert c["text"] == by_id[cid], (
            f"Returned text for chunk id={cid} does not match seeded text."
        )
        assert "embedding" not in c, (
            f"Returned chunk id={cid} leaks 'embedding' field; result must only contain "
            "{id, text, topic_id, tokens, score}."
        )
