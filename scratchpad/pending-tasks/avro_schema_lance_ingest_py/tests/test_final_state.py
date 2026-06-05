import importlib.util
import inspect
import os
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/avro_project"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
DB_DIR = os.path.join(PROJECT_DIR, "lance_db")
AVRO_PATH = "/app/data/records.avro"
RUN_ID = os.environ.get("ZEALT_RUN_ID", "local")
TABLE_NAME = f"records_{RUN_ID}"


def _load_solution_module():
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    assert spec is not None, f"Could not load spec for {SOLUTION_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_avro_rows():
    import fastavro

    with open(AVRO_PATH, "rb") as f:
        reader = fastavro.reader(f)
        rows = list(reader)
    return rows


@pytest.fixture(scope="module")
def cleaned_db():
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)
    yield


@pytest.fixture(scope="module")
def ingested(cleaned_db):
    assert os.path.isfile(SOLUTION_PATH), f"solution.py not found at {SOLUTION_PATH}"
    result = subprocess.run(
        ["python3", "solution.py", AVRO_PATH, TABLE_NAME],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"`python3 solution.py` failed with exit {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


@pytest.fixture(scope="module")
def avro_rows():
    return _read_avro_rows()


@pytest.fixture(scope="module")
def opened_table(ingested):
    import lancedb

    db = lancedb.connect(DB_DIR)
    assert TABLE_NAME in db.table_names(), (
        f"Table {TABLE_NAME!r} not found after ingest. "
        f"Existing tables: {db.table_names()}"
    )
    return db.open_table(TABLE_NAME)


def _author_field_path(schema):
    """Returns ('metadata.author', 'struct') or ('metadata_author', 'flat').

    Raises AssertionError if neither convention is present.
    """
    names = schema.names
    if "metadata" in names:
        meta_field = schema.field("metadata")
        assert pa.types.is_struct(meta_field.type), (
            f"`metadata` column is present but is not a struct: {meta_field.type}"
        )
        sub_names = [f.name for f in meta_field.type]
        for needed in ("author", "tags", "score"):
            assert needed in sub_names, (
                f"Nested metadata struct missing field {needed!r}. "
                f"Found: {sub_names}"
            )
        return "metadata.author", "struct"
    if {"metadata_author", "metadata_tags", "metadata_score"}.issubset(set(names)):
        return "metadata_author", "flat"
    raise AssertionError(
        "Schema must contain either a nested `metadata` struct or the flat "
        "columns metadata_author/metadata_tags/metadata_score. "
        f"Found columns: {names}"
    )


def test_solution_module_exists():
    assert os.path.isfile(SOLUTION_PATH), (
        f"Expected solution.py at {SOLUTION_PATH}"
    )


def test_solution_exposes_ingest_avro_callable():
    mod = _load_solution_module()
    assert hasattr(mod, "ingest_avro"), (
        "solution.py must define a top-level `ingest_avro` callable."
    )
    fn = getattr(mod, "ingest_avro")
    assert callable(fn), "`ingest_avro` must be callable."
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    assert len(params) >= 2, (
        f"`ingest_avro` must accept (avro_path, table_name); got signature {sig}."
    )


def test_cli_ingest_runs_successfully(ingested):
    # The fixture itself asserts exit code 0; this is a smoke check.
    assert ingested.returncode == 0


def test_lance_db_directory_created_after_ingest(ingested):
    assert os.path.isdir(DB_DIR), (
        f"LanceDB directory {DB_DIR} was not created by the ingest step."
    )


def test_table_row_count_matches_avro(opened_table, avro_rows):
    expected = len(avro_rows)
    actual = opened_table.count_rows()
    assert actual == expected, (
        f"Table row count {actual} does not match Avro record count {expected}."
    )
    assert expected == 300, (
        f"Avro fixture invariant: expected 300 records, found {expected}."
    )


def test_schema_id_and_title(opened_table):
    schema = opened_table.schema
    assert "id" in schema.names, f"Schema missing `id` column. Found {schema.names}"
    assert "title" in schema.names, (
        f"Schema missing `title` column. Found {schema.names}"
    )
    id_field = schema.field("id")
    title_field = schema.field("title")
    assert pa.types.is_int64(id_field.type), (
        f"`id` column expected int64, got {id_field.type}"
    )
    assert pa.types.is_string(title_field.type) or pa.types.is_large_string(
        title_field.type
    ), f"`title` column expected string, got {title_field.type}"


def test_schema_vector_column_is_fixed_size_list_float32_32(opened_table):
    schema = opened_table.schema
    assert "vector" in schema.names, (
        f"Schema missing `vector` column. Found {schema.names}"
    )
    field = schema.field("vector")
    t = field.type
    assert pa.types.is_fixed_size_list(t), (
        f"`vector` column must be fixed_size_list, got {t}"
    )
    assert t.list_size == 32, (
        f"`vector` column fixed_size_list size expected 32, got {t.list_size}"
    )
    assert pa.types.is_float32(t.value_type), (
        f"`vector` column value type expected float32, got {t.value_type}"
    )


def test_schema_metadata_struct_or_flat(opened_table):
    schema = opened_table.schema
    author_path, layout = _author_field_path(schema)
    if layout == "struct":
        meta_type = schema.field("metadata").type
        sub = {f.name: f.type for f in meta_type}
        assert pa.types.is_string(sub["author"]) or pa.types.is_large_string(
            sub["author"]
        ), f"metadata.author expected string, got {sub['author']}"
        assert pa.types.is_list(sub["tags"]) or pa.types.is_large_list(sub["tags"]), (
            f"metadata.tags expected list, got {sub['tags']}"
        )
        tags_value_type = sub["tags"].value_type
        assert pa.types.is_string(tags_value_type) or pa.types.is_large_string(
            tags_value_type
        ), f"metadata.tags items expected string, got {tags_value_type}"
        assert pa.types.is_float64(sub["score"]), (
            f"metadata.score expected float64, got {sub['score']}"
        )
    else:
        ma = schema.field("metadata_author").type
        mt = schema.field("metadata_tags").type
        ms = schema.field("metadata_score").type
        assert pa.types.is_string(ma) or pa.types.is_large_string(ma), (
            f"metadata_author expected string, got {ma}"
        )
        assert pa.types.is_list(mt) or pa.types.is_large_list(mt), (
            f"metadata_tags expected list, got {mt}"
        )
        mt_value_type = mt.value_type
        assert pa.types.is_string(mt_value_type) or pa.types.is_large_string(
            mt_value_type
        ), f"metadata_tags items expected string, got {mt_value_type}"
        assert pa.types.is_float64(ms), f"metadata_score expected float64, got {ms}"


def _brute_force_topk_l2(query, vectors, ids, k):
    diffs = vectors - query[None, :]
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    # Stable sort: ascending by distance, ties broken by id
    order = np.lexsort((np.asarray(ids), d2))
    return [int(ids[i]) for i in order[:k]]


def test_top5_l2_vector_search_matches_brute_force(opened_table, avro_rows):
    ids = [int(r["id"]) for r in avro_rows]
    vectors = np.array(
        [list(r["vector"]) for r in avro_rows], dtype=np.float32
    )
    q1 = np.random.default_rng(7).standard_normal(32).astype(np.float32)
    expected_ids = _brute_force_topk_l2(q1, vectors, ids, 5)

    rows = opened_table.search(q1).limit(5).to_list()
    actual_ids = [int(r["id"]) for r in rows]
    assert actual_ids == expected_ids, (
        f"Top-5 L2 search returned {actual_ids}, expected {expected_ids}."
    )


def test_filtered_vector_search_alice_matches_brute_force(opened_table, avro_rows):
    schema = opened_table.schema
    author_path, _ = _author_field_path(schema)

    target_author = "alice"
    alice_rows = [
        r for r in avro_rows if r["metadata"]["author"] == target_author
    ]
    assert len(alice_rows) > 5, (
        f"Avro fixture must contain >5 rows with author={target_author!r}. "
        f"Found {len(alice_rows)}."
    )
    ids = [int(r["id"]) for r in alice_rows]
    vectors = np.array(
        [list(r["vector"]) for r in alice_rows], dtype=np.float32
    )
    q2 = np.random.default_rng(11).standard_normal(32).astype(np.float32)
    expected_ids = _brute_force_topk_l2(q2, vectors, ids, 5)

    predicate = f"{author_path} = '{target_author}'"
    rows = (
        opened_table.search(q2)
        .where(predicate)
        .limit(5)
        .to_list()
    )
    actual_ids = [int(r["id"]) for r in rows]
    assert actual_ids == expected_ids, (
        f"Filtered top-5 L2 search (author={target_author!r}, predicate={predicate!r}) "
        f"returned {actual_ids}, expected {expected_ids}."
    )


def test_row_contents_preserved_for_top_hit(opened_table, avro_rows):
    ids = [int(r["id"]) for r in avro_rows]
    vectors = np.array(
        [list(r["vector"]) for r in avro_rows], dtype=np.float32
    )
    q1 = np.random.default_rng(7).standard_normal(32).astype(np.float32)
    expected_top = _brute_force_topk_l2(q1, vectors, ids, 1)[0]

    expected_row = next(r for r in avro_rows if int(r["id"]) == expected_top)

    rows = opened_table.search(q1).limit(1).to_list()
    assert rows, "Top-1 vector search returned no rows."
    row = rows[0]
    assert int(row["id"]) == expected_top, (
        f"Top-1 id mismatch: got {row['id']}, expected {expected_top}."
    )
    assert row["title"] == expected_row["title"], (
        f"Top-1 title mismatch: got {row['title']!r}, "
        f"expected {expected_row['title']!r}."
    )
    vec = list(row["vector"])
    assert len(vec) == 32, f"Top-1 vector length expected 32, got {len(vec)}"

    schema = opened_table.schema
    _, layout = _author_field_path(schema)
    expected_author = expected_row["metadata"]["author"]
    if layout == "struct":
        actual_author = row["metadata"]["author"]
    else:
        actual_author = row["metadata_author"]
    assert actual_author == expected_author, (
        f"Top-1 author mismatch: got {actual_author!r}, "
        f"expected {expected_author!r}."
    )
