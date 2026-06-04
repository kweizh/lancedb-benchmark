import json
import os

import pyarrow as pa
import pytest

import lancedb


OUTPUT_FILE = "/workspace/output/hybrid_rrf.json"
TABLE_NAME = "kb"


def _db_uri():
    return os.environ.get("LANCEDB_URI", "/workspace/db")


@pytest.fixture(scope="module")
def hybrid_results():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Hybrid search output file {OUTPUT_FILE} does not exist."
    )
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def kb_table():
    db = lancedb.connect(_db_uri())
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"Expected LanceDB table '{TABLE_NAME}' at {_db_uri()}, "
        f"but only found tables: {table_names}."
    )
    return db.open_table(TABLE_NAME)


def test_output_is_json_list_of_5(hybrid_results):
    assert isinstance(hybrid_results, list), (
        f"Expected the contents of {OUTPUT_FILE} to be a JSON array, "
        f"got type {type(hybrid_results).__name__}."
    )
    assert len(hybrid_results) == 5, (
        f"Expected exactly 5 hybrid search results in {OUTPUT_FILE}, "
        f"got {len(hybrid_results)}."
    )


def test_output_entries_have_id_text_pair(hybrid_results):
    for idx, entry in enumerate(hybrid_results):
        assert isinstance(entry, list) and len(entry) == 2, (
            f"Result #{idx} in {OUTPUT_FILE} must be a 2-element [id, text] list, "
            f"got {entry!r}."
        )
        row_id, text = entry
        assert isinstance(row_id, int) and not isinstance(row_id, bool), (
            f"Result #{idx} in {OUTPUT_FILE} must have an integer id, "
            f"got {type(row_id).__name__}: {row_id!r}."
        )
        assert isinstance(text, str), (
            f"Result #{idx} in {OUTPUT_FILE} must have a string text, "
            f"got {type(text).__name__}: {text!r}."
        )


def test_top_hit_contains_rrf_reranker_phrase(hybrid_results):
    top_id, top_text = hybrid_results[0]
    assert "rrf reranker" in top_text, (
        "The top hybrid + RRF result text must contain the phrase "
        f"'rrf reranker'; got id={top_id!r}, text={top_text!r}."
    )
    # The seeded dataset places this phrase exclusively in id=7 with the
    # exact text 'hybrid rrf reranker tutorial for lancedb'.
    assert top_id == 7, (
        "The top hybrid + RRF result must be the row whose text exactly "
        f"contains 'rrf reranker' (id=7). Got id={top_id!r}, text={top_text!r}."
    )
    assert top_text == "hybrid rrf reranker tutorial for lancedb", (
        "The top hybrid + RRF result text must equal the seeded row 7 text. "
        f"Got {top_text!r}."
    )


def test_table_has_at_least_30_rows(kb_table):
    n = kb_table.count_rows()
    assert n >= 30, (
        f"Expected table '{TABLE_NAME}' to contain at least 30 seeded rows, got {n}."
    )


def test_table_schema_has_string_text_column(kb_table):
    schema = kb_table.schema
    field_names = schema.names
    assert "text" in field_names, (
        f"Expected table '{TABLE_NAME}' to have a 'text' column; "
        f"got fields {field_names}."
    )
    text_field = schema.field("text")
    assert pa.types.is_string(text_field.type) or pa.types.is_large_string(text_field.type), (
        f"Expected 'text' column to be a string type, got {text_field.type}."
    )


def test_fts_index_exists_on_text(kb_table):
    indices = kb_table.list_indices()
    matching = []
    for idx in indices:
        cols = getattr(idx, "columns", None)
        if cols is None and isinstance(idx, dict):
            cols = idx.get("columns")
        if cols and "text" in cols:
            matching.append(idx)
    assert matching, (
        f"Expected at least one index on the 'text' column of table '{TABLE_NAME}' "
        f"(the native FTS index), got indices: {indices!r}."
    )
