"""Final-state verification for query_decomposition_rag_py.

Runs after the candidate finishes. Imports `solution` from /home/user/myproject
and exercises `decompose`, `decomposed_search`, and `baseline_search` against
the seeded 50-row LanceDB `docs` table.

The compound benchmark question targets three topics simultaneously
(Python GIL, Rust borrow checker, Go GC). A single-shot embedding tends to
collapse onto whichever facet has the strongest signal, so the topical
coverage of `baseline_search` is small; a working query-decomposition
pipeline that breaks the question into three sub-questions then unions the
per-sub-question retrieval should cover at least three distinct topics.
"""

import os
import sys

import lancedb
import numpy as np
import pytest
from openai import OpenAI

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
TABLE_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"

COMPOUND_QUESTION = (
    "What's the difference between Python's GIL and Rust's borrow checker, "
    "and how does Go's GC compare to both?"
)

TARGET_TOPICS = {"python_gil", "rust_borrow_checker", "go_gc"}


def _import_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import importlib

    if "solution" in sys.modules:
        importlib.reload(sys.modules["solution"])
    import solution  # type: ignore

    return solution


def _id_to_topic_map():
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()[["id", "topic"]]
    return {int(row["id"]): str(row["topic"]) for _, row in df.iterrows()}


def _embed(client, text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def test_solution_module_exposes_three_callables():
    solution = _import_solution()
    for name in ("decompose", "decomposed_search", "baseline_search"):
        fn = getattr(solution, name, None)
        assert callable(fn), f"solution.py must expose a callable `{name}`."


def test_decompose_returns_three_distinct_nonempty_strings():
    solution = _import_solution()
    subs = solution.decompose(COMPOUND_QUESTION)
    assert isinstance(subs, list), f"decompose must return a list; got {type(subs).__name__}."
    assert len(subs) == 3, f"decompose must return exactly 3 sub-questions; got {len(subs)}."
    for i, s in enumerate(subs):
        assert isinstance(s, str), f"sub-question {i} is not a str; got {type(s).__name__}."
        assert s.strip(), f"sub-question {i} is empty/whitespace."
    assert len({s.strip() for s in subs}) == 3, (
        f"decompose must return 3 distinct sub-questions; got duplicates in {subs!r}."
    )


def test_baseline_search_returns_top_k_ids():
    solution = _import_solution()
    ids = solution.baseline_search(COMPOUND_QUESTION, k=5)
    assert isinstance(ids, list), f"baseline_search must return a list; got {type(ids).__name__}."
    assert len(ids) == 5, f"baseline_search(k=5) must return 5 ids; got {len(ids)}."
    for x in ids:
        assert isinstance(x, (int, np.integer)), (
            f"baseline_search must return ints; got element of type {type(x).__name__}."
        )
    assert len(set(ids)) == 5, f"baseline_search returned duplicates: {ids!r}."


def test_baseline_search_matches_plain_lancedb_topk():
    """baseline_search must be a plain single-vector LanceDB search."""
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set in the verifier environment."

    solution = _import_solution()
    cand_ids = [int(x) for x in solution.baseline_search(COMPOUND_QUESTION, k=5)]

    client = OpenAI()
    qvec = _embed(client, COMPOUND_QUESTION)
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    rows = tbl.search(qvec).limit(5).to_list()
    truth_ids = [int(r["id"]) for r in rows]

    assert cand_ids == truth_ids, (
        f"baseline_search must replicate a plain top-5 LanceDB vector search of the question. "
        f"Got candidate={cand_ids!r}, expected={truth_ids!r}."
    )


def test_decomposed_search_topical_coverage_at_least_3():
    solution = _import_solution()
    ids = solution.decomposed_search(COMPOUND_QUESTION, k=5)
    assert isinstance(ids, list), f"decomposed_search must return a list; got {type(ids).__name__}."
    assert len(ids) == 5, f"decomposed_search(k=5) must return 5 ids; got {len(ids)}."
    assert len(set(ids)) == 5, f"decomposed_search returned duplicates: {ids!r}."

    id_to_topic = _id_to_topic_map()
    for x in ids:
        assert int(x) in id_to_topic, (
            f"decomposed_search returned id {x!r} that is not in the seeded `docs` table."
        )
    topics_in_top5 = {id_to_topic[int(x)] for x in ids}
    target_overlap = topics_in_top5 & TARGET_TOPICS
    assert len(target_overlap) >= 3, (
        f"decomposed_search top-5 must cover at least 3 distinct topics drawn from "
        f"{sorted(TARGET_TOPICS)!r}; got top-5 topics={sorted(topics_in_top5)!r}, "
        f"target overlap={sorted(target_overlap)!r}. A working decomposition pipeline "
        f"should retrieve evidence for all three of Python GIL, Rust borrow checker, "
        f"and Go GC. ids={ids!r}."
    )


def test_decomposed_outranks_baseline_in_topical_coverage():
    solution = _import_solution()
    dec_ids = solution.decomposed_search(COMPOUND_QUESTION, k=5)
    base_ids = solution.baseline_search(COMPOUND_QUESTION, k=5)

    id_to_topic = _id_to_topic_map()
    dec_topics = {id_to_topic[int(x)] for x in dec_ids}
    base_topics = {id_to_topic[int(x)] for x in base_ids}

    assert len(dec_topics) > len(base_topics), (
        f"decomposed_search top-5 must cover strictly more distinct topics than "
        f"baseline_search top-5. Got decomposed_topics={sorted(dec_topics)!r}, "
        f"baseline_topics={sorted(base_topics)!r}."
    )


def test_decomposed_ids_come_from_subquestion_top5_union():
    """Every id returned by decomposed_search must appear in some sub-question top-5.

    This proves the candidate actually unions per-sub-question retrieval rather
    than running a single shared embedding behind the scenes.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set in the verifier environment."

    solution = _import_solution()
    subs = solution.decompose(COMPOUND_QUESTION)
    assert isinstance(subs, list) and len(subs) == 3, (
        f"decompose must return exactly 3 sub-questions; got {subs!r}."
    )
    dec_ids = [int(x) for x in solution.decomposed_search(COMPOUND_QUESTION, k=5)]

    client = OpenAI()
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)

    union_ids = set()
    for sq in subs:
        qvec = _embed(client, sq)
        rows = tbl.search(qvec).limit(5).to_list()
        for r in rows:
            union_ids.add(int(r["id"]))

    missing = [x for x in dec_ids if x not in union_ids]
    assert not missing, (
        f"decomposed_search top-5 contains ids {missing!r} that are NOT in the union of "
        f"the per-sub-question top-5 LanceDB searches. Union={sorted(union_ids)!r}, "
        f"decomposed={dec_ids!r}."
    )
