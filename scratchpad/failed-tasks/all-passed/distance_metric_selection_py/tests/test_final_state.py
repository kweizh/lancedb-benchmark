import json
import os

import lancedb
import numpy as np
import pyarrow as pa
import pytest


OUTPUT_PATH = "/workspace/output/distances.json"
DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "vectors"
SEED = 123
NUM_ROWS = 32
VECTOR_DIM = 16
TOP_K = 5


def _expected_top_k():
    """Recompute the deterministic top-5 IDs for each distance metric."""
    rng = np.random.default_rng(SEED)
    data = rng.standard_normal(size=(NUM_ROWS, VECTOR_DIM)).astype("float32")
    query = rng.standard_normal(size=(VECTOR_DIM,)).astype("float32")

    # L2 (smaller is closer).
    l2_dists = np.linalg.norm(data - query, axis=1)
    l2_top = np.argsort(l2_dists, kind="stable")[:TOP_K].tolist()

    # Cosine distance = 1 - cosine similarity (smaller is closer).
    data_norms = np.linalg.norm(data, axis=1)
    query_norm = np.linalg.norm(query)
    cos_sim = (data @ query) / (data_norms * query_norm)
    cos_dists = 1.0 - cos_sim
    cosine_top = np.argsort(cos_dists, kind="stable")[:TOP_K].tolist()

    # Dot distance in LanceDB ranks by negative dot product (smaller is closer).
    dots = data @ query
    dot_dists = -dots
    dot_top = np.argsort(dot_dists, kind="stable")[:TOP_K].tolist()

    return {
        "l2": [int(x) for x in l2_top],
        "cosine": [int(x) for x in cosine_top],
        "dot": [int(x) for x in dot_top],
    }


@pytest.fixture(scope="module")
def output_json():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected output JSON at {OUTPUT_PATH}, but it does not exist."
    )
    with open(OUTPUT_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"Expected top-level JSON object in {OUTPUT_PATH}, got {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def table():
    db = lancedb.connect(DB_URI)
    names = db.table_names()
    assert TABLE_NAME in names, (
        f"Expected table '{TABLE_NAME}' in LanceDB at {DB_URI}, found tables: {names}."
    )
    return db.open_table(TABLE_NAME)


def test_output_json_has_exact_keys(output_json):
    assert set(output_json.keys()) == {"l2", "cosine", "dot"}, (
        f"Expected JSON keys to be exactly l2, cosine, dot; got {sorted(output_json.keys())}."
    )


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_output_json_value_shape(output_json, metric):
    values = output_json[metric]
    assert isinstance(values, list), (
        f"Expected JSON['{metric}'] to be a list, got {type(values).__name__}."
    )
    assert len(values) == TOP_K, (
        f"Expected JSON['{metric}'] to contain {TOP_K} entries, got {len(values)}."
    )
    for v in values:
        assert isinstance(v, int) and not isinstance(v, bool), (
            f"Expected each entry in JSON['{metric}'] to be a plain integer, got {type(v).__name__}: {v}."
        )


def test_table_row_count(table):
    assert table.count_rows() == NUM_ROWS, (
        f"Expected {NUM_ROWS} rows in table '{TABLE_NAME}', got {table.count_rows()}."
    )


def test_table_schema(table):
    schema = table.schema
    field_names = [f.name for f in schema]
    for required in ("id", "label", "vector"):
        assert required in field_names, (
            f"Expected field '{required}' in table schema, found {field_names}."
        )

    id_field = schema.field("id")
    assert pa.types.is_int64(id_field.type), (
        f"Expected 'id' to be int64, got {id_field.type}."
    )

    label_field = schema.field("label")
    assert pa.types.is_string(label_field.type) or pa.types.is_large_string(label_field.type), (
        f"Expected 'label' to be a string type, got {label_field.type}."
    )

    vector_field = schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type), (
        f"Expected 'vector' to be fixed_size_list, got {vector_field.type}."
    )
    assert vector_field.type.list_size == VECTOR_DIM, (
        f"Expected vector fixed_size_list size {VECTOR_DIM}, got {vector_field.type.list_size}."
    )
    assert pa.types.is_float32(vector_field.type.value_type), (
        f"Expected vector element type float32, got {vector_field.type.value_type}."
    )


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_top_k_matches_expected(output_json, metric):
    expected = _expected_top_k()[metric]
    actual = output_json[metric]
    assert actual == expected, (
        f"Top-{TOP_K} IDs for metric '{metric}' do not match expected. "
        f"Expected {expected}, got {actual}."
    )
