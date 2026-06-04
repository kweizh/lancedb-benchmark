import json
import os

import pytest


LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
RESULTS_PATH = "/workspace/output/fts_results.json"

EXPECTED_QUERIES = {
    "query_1": {"text": "vector database", "top_id": 1},
    "query_2": {"text": "lance format", "top_id": 2},
}


@pytest.fixture(scope="module")
def table():
    import lancedb

    db = lancedb.connect(LANCEDB_URI)
    names = db.table_names()
    assert "articles" in names, (
        f"Expected an 'articles' table at {LANCEDB_URI}; found tables={names!r}."
    )
    return db.open_table("articles")


def test_table_schema(table):
    import pyarrow as pa

    schema = table.schema
    field_names = {f.name for f in schema}
    for required in ("id", "title", "body", "vector"):
        assert required in field_names, (
            f"Column '{required}' missing from articles table; got fields={sorted(field_names)!r}."
        )

    id_field = schema.field("id")
    assert pa.types.is_int64(id_field.type), (
        f"Column 'id' must be int64; got {id_field.type!r}."
    )

    title_field = schema.field("title")
    assert pa.types.is_string(title_field.type) or pa.types.is_large_string(title_field.type), (
        f"Column 'title' must be a UTF8 string type; got {title_field.type!r}."
    )

    body_field = schema.field("body")
    assert pa.types.is_string(body_field.type) or pa.types.is_large_string(body_field.type), (
        f"Column 'body' must be a UTF8 string type; got {body_field.type!r}."
    )

    vector_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type), (
        f"Column 'vector' must be a fixed_size_list; got {vector_field.type!r}."
    )
    assert vector_field.type.list_size == 4, (
        f"Column 'vector' must have list size 4; got list_size={vector_field.type.list_size}."
    )
    value_type = vector_field.type.value_type
    assert pa.types.is_float32(value_type), (
        f"Column 'vector' values must be float32; got {value_type!r}."
    )


def test_row_count(table):
    n = table.count_rows()
    assert n >= 20, f"articles table must contain at least 20 rows; got {n}."


def test_fts_index_on_body(table):
    indices = list(table.list_indices())
    assert indices, "No indices found on the articles table; expected at least one FTS index on 'body'."

    matches = []
    for idx in indices:
        # Index objects expose `.columns` (list[str]) and `.index_type` (str-like).
        columns = getattr(idx, "columns", None)
        if columns is None and isinstance(idx, dict):
            columns = idx.get("columns")
        index_type = getattr(idx, "index_type", None)
        if index_type is None and isinstance(idx, dict):
            index_type = idx.get("index_type")
        index_type_str = str(index_type).upper() if index_type is not None else ""

        if columns and "body" in list(columns):
            matches.append((columns, index_type_str, idx))

    assert matches, (
        f"No index covering column 'body' was found. Existing indices: {indices!r}."
    )

    fts_matches = [m for m in matches if ("FTS" in m[1]) or ("INVERTED" in m[1])]
    assert fts_matches, (
        f"Expected an FTS / inverted index on 'body'; found indices on 'body' but none "
        f"were FTS. Details: {matches!r}."
    )

    # Native (non-Tantivy) FTS index — the index_type should not mention Tantivy.
    for cols, itype, _ in fts_matches:
        assert "TANTIVY" not in itype, (
            f"FTS index on 'body' must use the native (non-Tantivy) backend; got index_type={itype!r}."
        )


def test_results_file_exists():
    assert os.path.isfile(RESULTS_PATH), (
        f"Expected results JSON at {RESULTS_PATH}; file is missing."
    )


def test_results_json_structure():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"{RESULTS_PATH} must contain a JSON object at the top level; got {type(data).__name__}."
    )
    assert set(data.keys()) == {"query_1", "query_2"}, (
        f"{RESULTS_PATH} must contain exactly the keys 'query_1' and 'query_2'; got {sorted(data.keys())!r}."
    )

    for key in ("query_1", "query_2"):
        entries = data[key]
        assert isinstance(entries, list), (
            f"{RESULTS_PATH}: '{key}' must be a list; got {type(entries).__name__}."
        )
        assert 1 <= len(entries) <= 3, (
            f"{RESULTS_PATH}: '{key}' must contain 1-3 entries; got {len(entries)}."
        )
        scores = []
        for i, entry in enumerate(entries):
            assert isinstance(entry, dict), (
                f"{RESULTS_PATH}: '{key}'[{i}] must be a JSON object; got {type(entry).__name__}."
            )
            for required_key in ("id", "title", "_score"):
                assert required_key in entry, (
                    f"{RESULTS_PATH}: '{key}'[{i}] is missing required key '{required_key}'; entry={entry!r}."
                )
            assert isinstance(entry["id"], int), (
                f"{RESULTS_PATH}: '{key}'[{i}].id must be int; got {entry['id']!r}."
            )
            assert isinstance(entry["title"], str), (
                f"{RESULTS_PATH}: '{key}'[{i}].title must be str; got {entry['title']!r}."
            )
            assert isinstance(entry["_score"], (int, float)), (
                f"{RESULTS_PATH}: '{key}'[{i}]._score must be numeric; got {entry['_score']!r}."
            )
            scores.append(float(entry["_score"]))
        assert scores == sorted(scores, reverse=True), (
            f"{RESULTS_PATH}: '{key}' entries must be sorted by descending _score; got {scores!r}."
        )


def test_results_top_ids_match_ground_truth():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, expected in EXPECTED_QUERIES.items():
        top = data[key][0]
        assert top["id"] == expected["top_id"], (
            f"{RESULTS_PATH}: top result id for '{key}' (query='{expected['text']}') "
            f"must be {expected['top_id']}; got id={top['id']!r}, title={top.get('title')!r}."
        )


def test_independent_fts_query_matches_ground_truth(table):
    """Independently re-run the two FTS queries via the SDK and verify ground-truth top hits."""
    for key, expected in EXPECTED_QUERIES.items():
        hits = table.search(expected["text"], query_type="fts").limit(3).to_list()
        assert hits, (
            f"Independent FTS query for '{expected['text']}' returned no results."
        )
        assert int(hits[0]["id"]) == expected["top_id"], (
            f"Independent FTS query for '{expected['text']}' returned top id={hits[0]['id']!r} "
            f"(title={hits[0].get('title')!r}); expected id={expected['top_id']}."
        )
