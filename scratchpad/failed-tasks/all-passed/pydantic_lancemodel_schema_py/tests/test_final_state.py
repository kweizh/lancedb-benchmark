import json
import os
import re

import lancedb
import numpy as np
import pyarrow as pa
import pytest


PROJECT_DIR = "/workspace"
DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_DIR = "/workspace/output"
JSON_PATH = os.path.join(OUTPUT_DIR, "movies_top5.json")
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

EXPECTED_FIELDS = ["id", "title", "year", "genres", "summary", "vector"]
VECTOR_DIM = 32
NUM_ROWS = 24
RNG_SEED = 99


# ----------------------------- Helpers -----------------------------


def _compute_expected_top5_ids():
    rng = np.random.default_rng(RNG_SEED)
    matrix = np.stack(
        [rng.standard_normal(VECTOR_DIM).astype("float32") for _ in range(NUM_ROWS)],
        axis=0,
    )
    query = rng.standard_normal(VECTOR_DIM).astype("float32")
    dists = np.linalg.norm(matrix - query, axis=1)
    order = np.argsort(dists, kind="stable")[:5]
    return [int(i) for i in order.tolist()]


@pytest.fixture(scope="module")
def expected_top5_ids():
    return _compute_expected_top5_ids()


@pytest.fixture(scope="module")
def table():
    db = lancedb.connect(DB_URI)
    names = list(db.table_names())
    assert "movies" in names, (
        f"Expected table 'movies' in LanceDB at {DB_URI}, found tables: {names}."
    )
    return db.open_table("movies")


@pytest.fixture(scope="module")
def output_json():
    assert os.path.isfile(JSON_PATH), f"Output JSON not found at {JSON_PATH}."
    with open(JSON_PATH) as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def log_text():
    assert os.path.isfile(LOG_PATH), f"Log file not found at {LOG_PATH}."
    with open(LOG_PATH) as f:
        return f.read()


# ----------------------------- Tests -----------------------------


def test_output_files_exist():
    assert os.path.isfile(JSON_PATH), f"{JSON_PATH} does not exist."
    assert os.path.isfile(LOG_PATH), f"{LOG_PATH} does not exist."


def test_schema_field_names_order(table):
    schema = table.schema
    field_names = [f.name for f in schema]
    assert field_names == EXPECTED_FIELDS, (
        f"Expected schema field order {EXPECTED_FIELDS}, got {field_names}."
    )


def test_schema_field_types(table):
    schema = table.schema

    def _field(name):
        return schema.field(name)

    assert pa.types.is_int64(_field("id").type), (
        f"Field 'id' must be int64, got {_field('id').type}."
    )
    assert pa.types.is_string(_field("title").type) or pa.types.is_large_string(
        _field("title").type
    ), f"Field 'title' must be string/utf8, got {_field('title').type}."
    assert pa.types.is_int64(_field("year").type), (
        f"Field 'year' must be int64, got {_field('year').type}."
    )

    genres_type = _field("genres").type
    assert pa.types.is_list(genres_type) or pa.types.is_large_list(genres_type), (
        f"Field 'genres' must be a list type, got {genres_type}."
    )
    value_type = genres_type.value_type
    assert pa.types.is_string(value_type) or pa.types.is_large_string(value_type), (
        f"Field 'genres' must be list<string>, got list<{value_type}>."
    )

    assert pa.types.is_string(_field("summary").type) or pa.types.is_large_string(
        _field("summary").type
    ), f"Field 'summary' must be string/utf8, got {_field('summary').type}."

    vector_type = _field("vector").type
    assert pa.types.is_fixed_size_list(vector_type), (
        f"Field 'vector' must be fixed_size_list, got {vector_type}."
    )
    assert vector_type.list_size == VECTOR_DIM, (
        f"Field 'vector' must have size {VECTOR_DIM}, got {vector_type.list_size}."
    )
    assert pa.types.is_float32(vector_type.value_type), (
        f"Field 'vector' element type must be float32, got {vector_type.value_type}."
    )


def test_row_count(table):
    n = table.count_rows()
    assert n >= NUM_ROWS, f"Expected count_rows() >= {NUM_ROWS}, got {n}."


def test_output_json_shape(output_json):
    assert isinstance(output_json, list), "movies_top5.json must be a JSON array."
    assert len(output_json) == 5, (
        f"movies_top5.json must contain exactly 5 entries, got {len(output_json)}."
    )
    for i, row in enumerate(output_json):
        assert isinstance(row, dict), f"Entry {i} must be a JSON object."
        assert set(row.keys()) == {"id", "title", "year", "_distance"}, (
            f"Entry {i} must have exactly keys id, title, year, _distance; got {sorted(row.keys())}."
        )
        assert isinstance(row["id"], int) and not isinstance(row["id"], bool), (
            f"Entry {i}: 'id' must be int, got {type(row['id']).__name__}."
        )
        assert isinstance(row["title"], str), (
            f"Entry {i}: 'title' must be str, got {type(row['title']).__name__}."
        )
        assert isinstance(row["year"], int) and not isinstance(row["year"], bool), (
            f"Entry {i}: 'year' must be int, got {type(row['year']).__name__}."
        )
        assert isinstance(row["_distance"], float), (
            f"Entry {i}: '_distance' must be float, got {type(row['_distance']).__name__}."
        )


def test_output_json_sorted_by_distance(output_json):
    distances = [row["_distance"] for row in output_json]
    for a, b in zip(distances, distances[1:]):
        assert a <= b + 1e-6, (
            f"movies_top5.json must be ordered by ascending _distance, got {distances}."
        )


def test_output_json_scalar_fields_match_id_rule(output_json):
    for row in output_json:
        rid = row["id"]
        assert row["title"] == f"Movie {rid:02d}", (
            f"Entry id={rid}: expected title 'Movie {rid:02d}', got {row['title']!r}."
        )
        assert row["year"] == 2000 + (rid % 25), (
            f"Entry id={rid}: expected year {2000 + (rid % 25)}, got {row['year']}."
        )


def test_output_json_top5_ids_match_expected(output_json, expected_top5_ids):
    actual_ids = [row["id"] for row in output_json]
    assert actual_ids == expected_top5_ids, (
        f"Expected top-5 ids {expected_top5_ids} (recomputed from seeded RNG), "
        f"got {actual_ids}."
    )


def test_log_file_contains_top5_line(log_text, output_json):
    actual_ids = [row["id"] for row in output_json]
    match = re.search(r"Top-5 IDs:\s*(\[[^\]]*\])", log_text)
    assert match is not None, (
        f"Log file must contain a line like 'Top-5 IDs: <json-array>', got:\n{log_text}"
    )
    logged_ids = json.loads(match.group(1))
    assert logged_ids == actual_ids, (
        f"Log file 'Top-5 IDs' {logged_ids} must equal movies_top5.json ids {actual_ids}."
    )
