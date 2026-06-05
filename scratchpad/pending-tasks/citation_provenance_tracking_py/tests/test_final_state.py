import os
import sys
import importlib

import pytest

PROJECT_DIR = "/home/user/myproject"
SOURCE_DOCS_DIR = "/app/source_documents"
LANCEDB_DIR = "/app/lancedb_data"

QUERY_IN_CORPUS_1 = (
    "How does the photosynthesis process convert light into chemical energy?"
)
QUERY_IN_CORPUS_2 = "What are the major causes of World War I?"
QUERY_OFF_TOPIC = (
    "What is the best recipe for chocolate chip cookies with macadamia nuts?"
)


def _import_solution():
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    if "solution" in sys.modules:
        del sys.modules["solution"]
    return importlib.import_module("solution")


def _retrieve_top_k_triples(query: str, k: int) -> set:
    """Independently embed the query and run a vector search against the seeded table.

    Returns a set of (doc_id, span_start, span_end) triples for the top-k retrieved rows.
    """
    import lancedb
    from openai import OpenAI

    run_id = os.environ["ZEALT_RUN_ID"]
    client = OpenAI()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    qvec = resp.data[0].embedding

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(f"chunks_{run_id}")
    rows = (
        tbl.search(qvec)
        .distance_type("cosine")
        .limit(k)
        .to_list()
    )
    return {(r["doc_id"], int(r["span_start"]), int(r["span_end"])) for r in rows}


def _assert_quote_matches_source(cit: dict) -> None:
    doc_id = cit["doc_id"]
    span_start = int(cit["span_start"])
    span_end = int(cit["span_end"])
    src_path = os.path.join(SOURCE_DOCS_DIR, f"{doc_id}.txt")
    assert os.path.isfile(src_path), (
        f"Citation references missing source file: {src_path}"
    )
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert 0 <= span_start < span_end <= len(text), (
        f"Span [{span_start}:{span_end}] is out of bounds for {doc_id} "
        f"(len={len(text)})."
    )
    expected = text[span_start:span_end]
    assert cit["quote"] == expected, (
        f"Quote mismatch for {doc_id}[{span_start}:{span_end}]: "
        f"expected verbatim substring, got different content."
    )


def _assert_well_formed(result: dict) -> None:
    assert isinstance(result, dict), f"answer() must return a dict, got {type(result)}"
    assert set(result.keys()) == {"answer", "citations"}, (
        f"answer() must return keys exactly {{'answer','citations'}}, got {set(result.keys())}"
    )
    assert isinstance(result["answer"], str), "result['answer'] must be a string"
    assert isinstance(result["citations"], list), "result['citations'] must be a list"


def test_solution_module_importable():
    sol = _import_solution()
    assert hasattr(sol, "answer") and callable(sol.answer), (
        "solution.py must expose a callable `answer(query, k)`."
    )


def test_in_corpus_query_photosynthesis():
    sol = _import_solution()
    result = sol.answer(QUERY_IN_CORPUS_1, 5)
    _assert_well_formed(result)
    assert result["answer"] != "INSUFFICIENT_CONTEXT", (
        "Photosynthesis query should be answerable from the seeded corpus."
    )
    assert len(result["answer"]) > 20, (
        f"Answer text too short: {result['answer']!r}"
    )
    assert len(result["citations"]) >= 2, (
        f"Expected at least 2 citations, got {len(result['citations'])}"
    )
    for cit in result["citations"]:
        assert set(cit.keys()) == {"doc_id", "span_start", "span_end", "quote"}, (
            f"Citation has unexpected keys: {set(cit.keys())}"
        )
        _assert_quote_matches_source(cit)

    retrieved = _retrieve_top_k_triples(QUERY_IN_CORPUS_1, 5)
    for cit in result["citations"]:
        triple = (cit["doc_id"], int(cit["span_start"]), int(cit["span_end"]))
        assert triple in retrieved, (
            f"Citation triple {triple} is not in the top-5 retrieved set "
            f"{retrieved} — hallucinated chunk id."
        )


def test_in_corpus_query_world_war_one():
    sol = _import_solution()
    result = sol.answer(QUERY_IN_CORPUS_2, 5)
    _assert_well_formed(result)
    assert result["answer"] != "INSUFFICIENT_CONTEXT", (
        "World War I query should be answerable from the seeded corpus."
    )
    assert len(result["citations"]) >= 2, (
        f"Expected at least 2 citations, got {len(result['citations'])}"
    )
    for cit in result["citations"]:
        assert set(cit.keys()) == {"doc_id", "span_start", "span_end", "quote"}, (
            f"Citation has unexpected keys: {set(cit.keys())}"
        )
        _assert_quote_matches_source(cit)

    retrieved = _retrieve_top_k_triples(QUERY_IN_CORPUS_2, 5)
    for cit in result["citations"]:
        triple = (cit["doc_id"], int(cit["span_start"]), int(cit["span_end"]))
        assert triple in retrieved, (
            f"Citation triple {triple} not in top-5 retrieved set — hallucinated chunk id."
        )


def test_off_topic_query_returns_insufficient_context():
    sol = _import_solution()
    result = sol.answer(QUERY_OFF_TOPIC, 5)
    assert result == {"answer": "INSUFFICIENT_CONTEXT", "citations": []}, (
        f"Off-topic query must return the INSUFFICIENT_CONTEXT sentinel exactly, "
        f"got {result!r}"
    )


def test_provenance_is_subset_of_retrieval_on_repeat():
    sol = _import_solution()
    result = sol.answer(QUERY_IN_CORPUS_1, 5)
    _assert_well_formed(result)
    retrieved = _retrieve_top_k_triples(QUERY_IN_CORPUS_1, 5)
    cited = {
        (c["doc_id"], int(c["span_start"]), int(c["span_end"]))
        for c in result["citations"]
    }
    assert cited.issubset(retrieved), (
        f"On a repeat call, citation triples {cited} must remain a subset of the "
        f"top-5 retrieved set {retrieved}."
    )
