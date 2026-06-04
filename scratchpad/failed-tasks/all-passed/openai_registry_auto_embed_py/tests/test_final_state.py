import json
import os

import lancedb
import pyarrow as pa


WORKSPACE = "/workspace"
DEFAULT_DB_DIR = "/workspace/db"
OUTPUT_DIR = "/workspace/output"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "registry_results.json")
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

EXPECTED_TABLE = "docs"
EXPECTED_ROWS = 8
EXPECTED_DIM = 1536  # text-embedding-3-small
EXPECTED_TOP1_SUBSTR = "SQL filtering"


def _db_uri() -> str:
    return os.environ.get("LANCEDB_URI", DEFAULT_DB_DIR)


def test_run_log_exists_and_non_empty():
    assert os.path.isfile(LOG_PATH), f"Expected run log at {LOG_PATH} but it does not exist."
    assert os.path.getsize(LOG_PATH) > 0, f"Run log {LOG_PATH} exists but is empty."


def test_results_file_exists():
    assert os.path.isfile(RESULTS_PATH), (
        f"Expected results file at {RESULTS_PATH} but it does not exist."
    )


def test_results_file_is_valid_json_with_expected_shape():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict), "registry_results.json must be a JSON object."
    assert "top3_texts" in data, "registry_results.json missing key 'top3_texts'."
    assert "top3_labels" in data, "registry_results.json missing key 'top3_labels'."
    assert isinstance(data["top3_texts"], list), "'top3_texts' must be a list."
    assert isinstance(data["top3_labels"], list), "'top3_labels' must be a list."
    assert len(data["top3_texts"]) == 3, (
        f"Expected 'top3_texts' to have length 3, got {len(data['top3_texts'])}."
    )
    assert len(data["top3_labels"]) == 3, (
        f"Expected 'top3_labels' to have length 3, got {len(data['top3_labels'])}."
    )
    for i, t in enumerate(data["top3_texts"]):
        assert isinstance(t, str) and t, f"top3_texts[{i}] must be a non-empty string."
    for i, lab in enumerate(data["top3_labels"]):
        assert isinstance(lab, str) and lab, f"top3_labels[{i}] must be a non-empty string."


def test_top1_text_contains_sql_filtering_substring():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    top1 = data["top3_texts"][0]
    assert EXPECTED_TOP1_SUBSTR in top1, (
        f"Expected top-1 result text to contain substring '{EXPECTED_TOP1_SUBSTR}', "
        f"got: {top1!r}"
    )


def test_lancedb_table_exists_and_has_expected_row_count():
    db = lancedb.connect(_db_uri())
    table_names = db.table_names()
    assert EXPECTED_TABLE in table_names, (
        f"Expected table {EXPECTED_TABLE!r} in LanceDB at {_db_uri()}, "
        f"got tables: {table_names}"
    )
    tbl = db.open_table(EXPECTED_TABLE)
    n = tbl.count_rows()
    assert n == EXPECTED_ROWS, (
        f"Expected {EXPECTED_TABLE!r} to contain {EXPECTED_ROWS} rows, got {n}."
    )


def test_lancedb_vector_column_has_expected_dimension():
    db = lancedb.connect(_db_uri())
    tbl = db.open_table(EXPECTED_TABLE)
    schema = tbl.schema
    field_names = [f.name for f in schema]
    assert "vector" in field_names, (
        f"Expected a 'vector' column in {EXPECTED_TABLE!r} schema, got fields: {field_names}"
    )
    vec_field = schema.field("vector")
    vec_type = vec_field.type
    assert isinstance(vec_type, pa.FixedSizeListType), (
        f"Expected 'vector' field to be a FixedSizeList, got {vec_type!r}. "
        "This is what Vector(func.ndims()) + VectorField() produces."
    )
    assert vec_type.list_size == EXPECTED_DIM, (
        f"Expected vector dim {EXPECTED_DIM} (text-embedding-3-small), "
        f"got list_size={vec_type.list_size}."
    )
    value_type = vec_type.value_type
    assert pa.types.is_floating(value_type), (
        f"Expected vector value type to be a floating-point type, got {value_type!r}."
    )
