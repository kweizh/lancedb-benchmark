"""Initial-state verification for the query_decomposition_rag_py task.

Runs before the candidate touches the project. Confirms that the pre-seeded
50-row LanceDB `docs` table (5 topics x 10 docs, 1536-d real OpenAI
embeddings) is already present in the container image.
"""

import os

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb")
TABLE_NAME = "docs"
EMBED_DIM = 1536
EXPECTED_TOPICS = {
    "python_gil",
    "rust_borrow_checker",
    "go_gc",
    "javascript_event_loop",
    "java_jit",
}


def test_lancedb_importable():
    import lancedb  # noqa: F401


def test_openai_importable():
    import openai  # noqa: F401


def test_pyarrow_importable():
    import pyarrow  # noqa: F401


def test_project_dir_exists():
    assert os.path.isdir(PROJECT_DIR), f"Project directory {PROJECT_DIR} does not exist."


def test_lancedb_dir_exists():
    assert os.path.isdir(DB_DIR), f"LanceDB directory {DB_DIR} does not exist."


def test_docs_table_present_with_50_rows():
    import lancedb

    db = lancedb.connect(DB_DIR)
    names = db.table_names()
    assert TABLE_NAME in names, (
        f"Table {TABLE_NAME!r} not found in LanceDB at {DB_DIR}; got {names!r}."
    )
    tbl = db.open_table(TABLE_NAME)
    n = tbl.count_rows()
    assert n == 50, f"Expected exactly 50 seeded rows in {TABLE_NAME!r}; got {n}."


def test_docs_schema_has_embedding_dim_and_topic():
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    schema = tbl.schema
    field_names = set(schema.names)
    for key in ("id", "topic", "content", "embedding"):
        assert key in field_names, f"Schema is missing column {key!r}; got {field_names!r}."
    emb_field = schema.field("embedding")
    assert emb_field.type.list_size == EMBED_DIM, (
        f"Expected embedding column to be a fixed-size list of length {EMBED_DIM}; "
        f"got list_size={emb_field.type.list_size}."
    )


def test_topics_balanced_5x10():
    import lancedb

    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(TABLE_NAME)
    rows = tbl.to_pandas()
    topics_present = set(rows["topic"].unique().tolist())
    assert topics_present == EXPECTED_TOPICS, (
        f"Seeded topics differ from expected. Expected {EXPECTED_TOPICS!r}; got {topics_present!r}."
    )
    counts = rows.groupby("topic").size().to_dict()
    for t in EXPECTED_TOPICS:
        assert counts.get(t, 0) == 10, (
            f"Expected exactly 10 documents for topic {t!r}; got {counts.get(t, 0)}."
        )
