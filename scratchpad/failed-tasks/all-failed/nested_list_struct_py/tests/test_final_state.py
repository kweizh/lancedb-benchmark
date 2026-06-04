import json
import os

import lancedb
import numpy as np
import pyarrow as pa
import pytest


DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "papers"
OUTPUT_FILE = "/workspace/output/nested_results.json"

# Fixed query vector used by the candidate AND the verifier.
QUERY_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Ground-truth row data, mirroring the spec in task.json's `truth` field.
EXPECTED_ROWS = [
    {"id": 1, "title": "Paper One", "author0": "alice", "year": 2019, "citations": 12},
    {"id": 2, "title": "Paper Two", "author0": "bob", "year": 2020, "citations": 25},
    {"id": 3, "title": "Paper Three", "author0": "carol", "year": 2021, "citations": 7},
    {"id": 4, "title": "Paper Four", "author0": "dan", "year": 2022, "citations": 41},
    {"id": 5, "title": "Paper Five", "author0": "eve", "year": 2022, "citations": 3},
    {"id": 6, "title": "Paper Six", "author0": "frank", "year": 2023, "citations": 88},
    {"id": 7, "title": "Paper Seven", "author0": "grace", "year": 2023, "citations": 15},
    {"id": 8, "title": "Paper Eight", "author0": "heidi", "year": 2024, "citations": 60},
    {"id": 9, "title": "Paper Nine", "author0": "ivan", "year": 2024, "citations": 2},
    {"id": 10, "title": "Paper Ten", "author0": "judy", "year": 2025, "citations": 33},
]


def _expected_vector(row_id: int) -> np.ndarray:
    """Deterministic 8-d float32 vector for a given row id."""
    rng = np.random.default_rng(1000 + row_id)
    return rng.random(8, dtype=np.float32)


@pytest.fixture(scope="module")
def table():
    db = lancedb.connect(DB_URI)
    assert TABLE_NAME in db.table_names(), (
        f"Expected table '{TABLE_NAME}' to exist in LanceDB at {DB_URI}, "
        f"got tables: {db.table_names()}"
    )
    return db.open_table(TABLE_NAME)


@pytest.fixture(scope="module")
def result_json():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Expected output JSON at {OUTPUT_FILE}, but file does not exist."
    )
    with open(OUTPUT_FILE) as f:
        data = json.load(f)
    return data


def test_output_json_keys(result_json):
    assert isinstance(result_json, dict), (
        f"Expected nested_results.json to be a JSON object, got {type(result_json)}."
    )
    assert "topk_titles" in result_json, (
        "Expected 'topk_titles' key in nested_results.json."
    )
    assert "recent_ids_sorted" in result_json, (
        "Expected 'recent_ids_sorted' key in nested_results.json."
    )


def test_schema_matches_spec(table):
    schema = table.schema
    field_names = schema.names
    for expected in ["id", "title", "authors", "metrics", "vector"]:
        assert expected in field_names, (
            f"Expected column '{expected}' in table schema, got: {field_names}"
        )

    id_type = schema.field("id").type
    assert id_type == pa.int64(), f"Expected id: int64, got {id_type}."

    title_type = schema.field("title").type
    assert title_type == pa.string(), f"Expected title: string, got {title_type}."

    authors_type = schema.field("authors").type
    assert pa.types.is_list(authors_type), (
        f"Expected authors to be a list type, got {authors_type}."
    )
    inner = authors_type.value_type
    assert pa.types.is_struct(inner), (
        f"Expected authors list element to be a struct, got {inner}."
    )
    inner_fields = {inner.field(i).name: inner.field(i).type for i in range(inner.num_fields)}
    assert inner_fields.get("name") == pa.string(), (
        f"Expected authors.name: string, got {inner_fields.get('name')}."
    )
    assert inner_fields.get("affiliation") == pa.string(), (
        f"Expected authors.affiliation: string, got {inner_fields.get('affiliation')}."
    )

    metrics_type = schema.field("metrics").type
    assert pa.types.is_struct(metrics_type), (
        f"Expected metrics to be a struct, got {metrics_type}."
    )
    metric_fields = {
        metrics_type.field(i).name: metrics_type.field(i).type
        for i in range(metrics_type.num_fields)
    }
    assert metric_fields.get("citations") == pa.int32(), (
        f"Expected metrics.citations: int32, got {metric_fields.get('citations')}."
    )
    assert metric_fields.get("year") == pa.int32(), (
        f"Expected metrics.year: int32, got {metric_fields.get('year')}."
    )

    vector_type = schema.field("vector").type
    assert pa.types.is_fixed_size_list(vector_type), (
        f"Expected vector to be a fixed_size_list, got {vector_type}."
    )
    assert vector_type.list_size == 8, (
        f"Expected vector list_size to be 8, got {vector_type.list_size}."
    )
    assert vector_type.value_type == pa.float32(), (
        f"Expected vector value_type to be float32, got {vector_type.value_type}."
    )


def test_row_count(table):
    n = table.count_rows()
    assert n == 10, f"Expected exactly 10 rows in '{TABLE_NAME}', got {n}."


def test_topk_titles_match_recomputed(table, result_json):
    """Recompute the expected top-3 titles by running the same vector search."""
    results = table.search(QUERY_VECTOR).limit(3).to_list()
    expected_titles = [r["title"] for r in results]
    actual_titles = result_json["topk_titles"]
    assert isinstance(actual_titles, list), (
        f"Expected topk_titles to be a list, got {type(actual_titles)}."
    )
    assert len(actual_titles) == 3, (
        f"Expected topk_titles to have 3 elements, got {len(actual_titles)}."
    )
    assert actual_titles == expected_titles, (
        f"Expected topk_titles {expected_titles}, got {actual_titles}."
    )


def test_recent_ids_match_struct_filter(table, result_json):
    """Use a nested struct-subfield filter to derive expected ids."""
    rows = (
        table.search()
        .where("metrics.year >= 2022")
        .limit(50)
        .to_list()
    )
    expected_ids = sorted(r["id"] for r in rows)
    actual_ids = result_json["recent_ids_sorted"]
    assert isinstance(actual_ids, list), (
        f"Expected recent_ids_sorted to be a list, got {type(actual_ids)}."
    )
    assert actual_ids == sorted(actual_ids), (
        f"Expected recent_ids_sorted to be sorted ascending, got {actual_ids}."
    )
    assert actual_ids == expected_ids, (
        f"Expected recent_ids_sorted {expected_ids}, got {actual_ids}."
    )
    # Cross-check against the hard-coded ground truth.
    ground_truth_ids = sorted(
        r["id"] for r in EXPECTED_ROWS if r["year"] >= 2022
    )
    assert expected_ids == ground_truth_ids, (
        f"Expected struct filter to yield ids {ground_truth_ids}, got {expected_ids}."
    )


def test_recent_id_values_correspond_to_recent_years(table, result_json):
    """Direct row inspection: every id in recent_ids_sorted must have year >= 2022."""
    all_rows = table.to_pandas()
    id_to_year = {}
    for _, row in all_rows.iterrows():
        metrics = row["metrics"]
        if isinstance(metrics, dict):
            year = metrics.get("year")
        else:
            # struct may surface as a structured numpy scalar / namedtuple-like.
            year = metrics["year"]
        id_to_year[int(row["id"])] = int(year)
    for rid in result_json["recent_ids_sorted"]:
        assert rid in id_to_year, f"recent id {rid} not found in table."
        assert id_to_year[rid] >= 2022, (
            f"recent id {rid} has year {id_to_year[rid]} < 2022."
        )
