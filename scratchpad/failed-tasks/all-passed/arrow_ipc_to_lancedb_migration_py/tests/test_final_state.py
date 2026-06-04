import json
import os
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

import lancedb

PROJECT_DIR = "/home/user/myproject"
RUN_SCRIPT = os.path.join(PROJECT_DIR, "run.py")
LANCE_DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
SOURCE_PATH = "/app/source/dataset.arrows"
QUERY_VECTOR_PATH = "/app/query_vector.npy"


def _run_id():
    val = os.environ.get("ZEALT_RUN_ID")
    assert val, "ZEALT_RUN_ID environment variable must be set for verification."
    return val


def _load_source_table():
    with pa.OSFile(SOURCE_PATH, "rb") as sink:
        with ipc.open_stream(sink) as reader:
            return reader.read_all()


def _expected_top5(source_table, query_vector):
    embeddings = np.asarray(source_table.column("embedding").to_pylist(), dtype=np.float32)
    ids = np.asarray(source_table.column("id").to_pylist(), dtype=np.int64)
    diffs = embeddings - query_vector.reshape(1, -1).astype(np.float32)
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    order = np.argsort(dists, kind="stable")[:5]
    return [int(ids[i]) for i in order], [float(dists[i]) for i in order]


@pytest.fixture(scope="session")
def candidate_output():
    """Clean DB dir, run the candidate's script, return parsed stdout JSON."""
    if os.path.isdir(LANCE_DB_DIR):
        subprocess.run(["rm", "-rf", LANCE_DB_DIR], check=True)

    assert os.path.isfile(RUN_SCRIPT), (
        f"Candidate must provide an executable entrypoint at {RUN_SCRIPT}."
    )

    env = os.environ.copy()
    result = subprocess.run(
        ["python3", RUN_SCRIPT],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        "Candidate script python3 run.py exited non-zero.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    stdout = result.stdout.strip()
    # Tolerate extra whitespace; the candidate must produce a JSON object as the
    # main payload. Find the last JSON object in stdout.
    payload = None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        # Try to recover the final JSON object on the last non-empty line.
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    assert payload is not None, (
        "Candidate stdout did not contain a parseable JSON object. stdout was:\n"
        + stdout
    )
    return payload


@pytest.fixture(scope="session")
def source_table():
    return _load_source_table()


@pytest.fixture(scope="session")
def query_vector():
    return np.load(QUERY_VECTOR_PATH).astype(np.float32)


def test_stdout_json_shape(candidate_output):
    assert isinstance(candidate_output, dict), (
        "Candidate stdout JSON must be an object."
    )
    for key in ("table_name", "row_count", "schema_match", "top5"):
        assert key in candidate_output, (
            f"Candidate stdout JSON is missing required key '{key}'. Got keys: "
            f"{list(candidate_output.keys())}"
        )


def test_table_name_uses_run_id(candidate_output):
    expected = f"events_{_run_id()}"
    assert candidate_output["table_name"] == expected, (
        f"Expected table_name '{expected}', got {candidate_output['table_name']!r}."
    )


def test_schema_match_is_true(candidate_output):
    assert candidate_output["schema_match"] is True, (
        f"Expected schema_match to be JSON true; got {candidate_output['schema_match']!r}."
    )


def test_row_count_matches_source(candidate_output, source_table):
    assert candidate_output["row_count"] == source_table.num_rows, (
        f"row_count must equal source row count {source_table.num_rows}; "
        f"got {candidate_output['row_count']}."
    )


def test_top5_shape(candidate_output):
    top5 = candidate_output["top5"]
    assert isinstance(top5, list) and len(top5) == 5, (
        f"top5 must be a length-5 list; got {top5!r}."
    )
    for i, hit in enumerate(top5):
        assert isinstance(hit, dict), f"top5[{i}] must be an object."
        assert "id" in hit and "distance" in hit, (
            f"top5[{i}] must have keys 'id' and 'distance'; got {hit!r}."
        )
        assert isinstance(hit["id"], int), (
            f"top5[{i}].id must be an integer; got {hit['id']!r}."
        )
        assert isinstance(hit["distance"], (int, float)), (
            f"top5[{i}].distance must be numeric; got {hit['distance']!r}."
        )


def test_top5_ids_match_ground_truth(candidate_output, source_table, query_vector):
    expected_ids, _ = _expected_top5(source_table, query_vector)
    actual_ids = [hit["id"] for hit in candidate_output["top5"]]
    assert actual_ids == expected_ids, (
        "Candidate top-5 IDs do not match independent ground truth.\n"
        f"expected: {expected_ids}\nactual:   {actual_ids}"
    )


def test_lancedb_table_exists_and_row_count():
    db = lancedb.connect(LANCE_DB_DIR)
    table_name = f"events_{_run_id()}"
    names = list(db.table_names())
    assert table_name in names, (
        f"Expected LanceDB to contain table '{table_name}'; got {names}."
    )
    tbl = db.open_table(table_name)
    assert tbl.count_rows() == 2000, (
        f"Destination table must have 2000 rows; got {tbl.count_rows()}."
    )


def test_lancedb_table_schema_matches_source(source_table):
    db = lancedb.connect(LANCE_DB_DIR)
    table_name = f"events_{_run_id()}"
    tbl = db.open_table(table_name)
    dst_schema = tbl.schema
    src_schema = source_table.schema

    src_fields = {f.name: f.type for f in src_schema}
    dst_fields = {f.name: f.type for f in dst_schema}
    for name, src_type in src_fields.items():
        assert name in dst_fields, (
            f"Destination schema is missing column '{name}'."
        )
        assert dst_fields[name].equals(src_type), (
            f"Column '{name}' has type {dst_fields[name]} in destination but "
            f"{src_type} in source."
        )

    emb_type = dst_fields["embedding"]
    assert pa.types.is_fixed_size_list(emb_type), (
        f"Destination 'embedding' must be fixed_size_list; got {emb_type}."
    )
    assert emb_type.list_size == 48, (
        f"Destination 'embedding' width must be 48; got {emb_type.list_size}."
    )
    assert pa.types.is_float32(emb_type.value_type), (
        f"Destination 'embedding' value type must be float32; got {emb_type.value_type}."
    )


def test_lancedb_search_matches_ground_truth(source_table, query_vector):
    expected_ids, _ = _expected_top5(source_table, query_vector)
    db = lancedb.connect(LANCE_DB_DIR)
    table_name = f"events_{_run_id()}"
    tbl = db.open_table(table_name)
    hits = tbl.search(query_vector).limit(5).to_list()
    actual_ids = [int(h["id"]) for h in hits]
    assert actual_ids == expected_ids, (
        "LanceDB search on the migrated table does not return the ground-truth "
        f"top-5 IDs.\nexpected: {expected_ids}\nactual:   {actual_ids}"
    )
