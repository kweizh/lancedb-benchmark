"""Initial-state verification for the hf_inference_cross_encoder_rerank_py task.

These tests run BEFORE the candidate touches the project. They check that the
LanceDB database, the seeded `docs` table (200 rows, real OpenAI 1536-d
embeddings) and the verifier fixture `/home/user/myproject/.expected.json` are
already in place inside the container image.
"""

import json
import os

import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
EXPECTED_JSON = os.path.join(PROJECT_DIR, ".expected.json")
TABLE_NAME = "docs"
EMBED_DIM = 1536


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_openai_importable():
    import openai  # noqa: F401


def test_httpx_importable():
    import httpx  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(DB_DIR), f"LanceDB directory {DB_DIR} does not exist."


def test_expected_json_exists():
    assert os.path.isfile(EXPECTED_JSON), f"Fixture file {EXPECTED_JSON} does not exist."


def test_expected_json_has_required_keys():
    with open(EXPECTED_JSON) as f:
        payload = json.load(f)
    for key in ("anchor_query", "rigged_correct_id", "rigged_distractor_id"):
        assert key in payload, f"Fixture file missing required key {key!r}."
    assert payload["rigged_correct_id"] == "rigged-correct"
    assert payload["rigged_distractor_id"] == "rigged-distractor"
    assert isinstance(payload["anchor_query"], str) and payload["anchor_query"]


def test_docs_table_present_with_200_rows():
    import lancedb

    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert TABLE_NAME in names, f"Table {TABLE_NAME!r} not found in LanceDB at {DB_DIR}; got {names!r}."
    tbl = db.open_table(TABLE_NAME)
    n = tbl.count_rows()
    assert n == 200, f"Expected exactly 200 seeded rows in {TABLE_NAME!r}; got {n}."


def test_docs_schema_has_embedding_dim():
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    schema = tbl.schema
    field_names = set(schema.names)
    for key in ("id", "content", "embedding"):
        assert key in field_names, f"Schema is missing column {key!r}; got {field_names!r}."
    emb_field = schema.field("embedding")
    # fixed_size_list<float, 1536>
    assert emb_field.type.list_size == EMBED_DIM, (
        f"Expected embedding column to be a fixed-size list of length {EMBED_DIM}; "
        f"got list_size={emb_field.type.list_size}."
    )


def test_rigged_docs_present():
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.search().where("id = 'rigged-correct' OR id = 'rigged-distractor'").limit(10).to_list()
    ids = {row["id"] for row in df}
    assert ids == {"rigged-correct", "rigged-distractor"}, (
        f"Expected both rigged sentinel docs present; got {ids!r}."
    )
