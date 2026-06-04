import json
import os

import lancedb


LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
ARTIFACT_PATH = "/workspace/output/ivf_pq.json"
TABLE_NAME = "embeddings"
INDEX_NAME = "vector_idx"


def _load_artifact():
    assert os.path.isfile(ARTIFACT_PATH), (
        f"Expected output artifact at {ARTIFACT_PATH} but it does not exist."
    )
    with open(ARTIFACT_PATH, "r") as f:
        return json.load(f)


def _open_table():
    db = lancedb.connect(LANCEDB_URI)
    table_names = list(db.table_names())
    assert TABLE_NAME in table_names, (
        f"Expected table '{TABLE_NAME}' to exist in LanceDB at {LANCEDB_URI}; "
        f"found tables: {table_names}"
    )
    return db.open_table(TABLE_NAME)


def test_artifact_is_valid_json_with_expected_keys():
    data = _load_artifact()
    assert isinstance(data, dict), "Artifact JSON root must be an object."
    for key in ("index_present", "num_indexed_rows", "topk_ids"):
        assert key in data, f"Artifact JSON is missing required key '{key}'."


def test_index_present_flag_is_true():
    data = _load_artifact()
    assert data["index_present"] is True, (
        f"Expected 'index_present' to be true, got {data['index_present']!r}."
    )


def test_ivf_pq_index_exists_on_vector_column():
    table = _open_table()
    indices = table.list_indices()
    # list_indices() returns a list of IndexConfig-like objects with .name,
    # .index_type, and .columns attributes in lancedb 0.25.3.
    matching = []
    for idx in indices:
        index_type = getattr(idx, "index_type", None) or (
            idx.get("index_type") if isinstance(idx, dict) else None
        )
        columns = getattr(idx, "columns", None) or (
            idx.get("columns") if isinstance(idx, dict) else None
        )
        if index_type is None or columns is None:
            continue
        if str(index_type).upper().endswith("IVF_PQ") or "IVF_PQ" in str(index_type).upper():
            if "vector" in list(columns):
                matching.append(idx)
    assert matching, (
        f"Expected at least one IVF_PQ index on column 'vector'; "
        f"list_indices() returned: {indices}"
    )


def test_num_indexed_rows_matches_stats_and_meets_threshold():
    data = _load_artifact()
    table = _open_table()
    stats = table.index_stats(INDEX_NAME)
    assert stats is not None, (
        f"index_stats('{INDEX_NAME}') returned None; index may not have been created."
    )
    actual = getattr(stats, "num_indexed_rows", None)
    if actual is None and isinstance(stats, dict):
        actual = stats.get("num_indexed_rows")
    assert isinstance(actual, int), (
        f"Expected num_indexed_rows to be an int in index_stats output, got {actual!r}."
    )
    assert actual >= 256, (
        f"Expected at least 256 indexed rows for IVF_PQ training; got {actual}."
    )

    reported = data["num_indexed_rows"]
    assert isinstance(reported, int), (
        f"Expected JSON 'num_indexed_rows' to be int, got {type(reported).__name__}."
    )
    assert reported >= 256, (
        f"Expected JSON 'num_indexed_rows' >= 256, got {reported}."
    )
    assert reported == actual, (
        f"JSON 'num_indexed_rows' ({reported}) does not match index_stats "
        f"value ({actual})."
    )


def test_topk_ids_shape_and_range():
    data = _load_artifact()
    topk = data["topk_ids"]
    assert isinstance(topk, list), (
        f"Expected 'topk_ids' to be a list, got {type(topk).__name__}."
    )
    assert len(topk) == 10, (
        f"Expected exactly 10 ids in 'topk_ids', got {len(topk)}."
    )
    for v in topk:
        assert isinstance(v, int) and not isinstance(v, bool), (
            f"Each entry in 'topk_ids' must be an int, got {v!r}."
        )

    # IDs must lie in the seeded range [1, num_rows].
    table = _open_table()
    num_rows = table.count_rows()
    assert num_rows >= 512, (
        f"Expected at least 512 rows in '{TABLE_NAME}', got {num_rows}."
    )
    for v in topk:
        assert 1 <= v <= num_rows, (
            f"Id {v} in 'topk_ids' is outside the valid range [1, {num_rows}]."
        )

    assert len(set(topk)) == len(topk), (
        f"'topk_ids' must not contain duplicates; got {topk}."
    )
