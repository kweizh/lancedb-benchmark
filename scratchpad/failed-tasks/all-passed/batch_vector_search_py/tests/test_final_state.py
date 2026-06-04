import json
import os

import numpy as np
import lancedb


OUTPUT_FILE = "/workspace/output/batch_search.json"
DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "items"
NUM_ROWS = 64
VECTOR_DIM = 12
NUM_QUERIES = 5
TOP_K = 3
SEED = 33


def _expected_top_k_ids():
    """Recompute the expected top-K ids per query from the deterministic RNG sequence."""
    rng = np.random.default_rng(SEED)
    stored = np.stack(
        [rng.random(VECTOR_DIM, dtype=np.float32) for _ in range(NUM_ROWS)],
        axis=0,
    )  # shape (NUM_ROWS, VECTOR_DIM)
    queries = np.stack(
        [rng.random(VECTOR_DIM, dtype=np.float32) for _ in range(NUM_QUERIES)],
        axis=0,
    )  # shape (NUM_QUERIES, VECTOR_DIM)

    expected = []
    for q_idx in range(NUM_QUERIES):
        diff = stored - queries[q_idx]
        dists = np.sum(diff * diff, axis=1)
        # Use a stable sort so equal-distance ties resolve by ascending id, matching
        # LanceDB's in-insertion-order behavior for unindexed/flat tables.
        top_ids = np.argsort(dists, kind="stable")[:TOP_K].tolist()
        expected.append([int(i) for i in top_ids])
    return expected


def test_output_file_exists():
    assert os.path.isfile(OUTPUT_FILE), (
        f"Expected the candidate to produce {OUTPUT_FILE}, but it does not exist."
    )


def test_output_file_is_valid_json_object():
    with open(OUTPUT_FILE) as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"{OUTPUT_FILE} must contain a JSON object, got {type(data).__name__}."
    )
    assert "results" in data and len(data) == 1, (
        f"{OUTPUT_FILE} must contain exactly one top-level key 'results'; got keys {list(data.keys())}."
    )


def test_results_shape():
    with open(OUTPUT_FILE) as f:
        data = json.load(f)
    results = data["results"]
    assert isinstance(results, list), (
        f"'results' must be a list, got {type(results).__name__}."
    )
    assert len(results) == NUM_QUERIES, (
        f"'results' must have exactly {NUM_QUERIES} entries (one per query), got {len(results)}."
    )
    for i, row in enumerate(results):
        assert isinstance(row, list), (
            f"results[{i}] must be a list, got {type(row).__name__}."
        )
        assert len(row) == TOP_K, (
            f"results[{i}] must contain exactly {TOP_K} ids, got {len(row)}."
        )
        for j, v in enumerate(row):
            assert isinstance(v, int) and not isinstance(v, bool), (
                f"results[{i}][{j}] must be an integer id, got {type(v).__name__}={v!r}."
            )


def test_results_match_expected_top_k():
    with open(OUTPUT_FILE) as f:
        data = json.load(f)
    actual = data["results"]
    expected = _expected_top_k_ids()
    assert actual == expected, (
        f"Per-query top-{TOP_K} ids do not match expected.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )


def test_lancedb_table_seeded_with_64_rows():
    db = lancedb.connect(DB_URI)
    assert TABLE_NAME in db.table_names(), (
        f"Expected LanceDB table '{TABLE_NAME}' at {DB_URI}; "
        f"found tables: {db.table_names()}."
    )
    tbl = db.open_table(TABLE_NAME)
    n = tbl.count_rows()
    assert n == NUM_ROWS, (
        f"Expected LanceDB table '{TABLE_NAME}' to contain {NUM_ROWS} rows, got {n}."
    )
