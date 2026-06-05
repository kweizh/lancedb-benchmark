import json
import math
import os

import lancedb
import pyarrow as pa
import pytest

PROJECT_DIR = "/home/user/myproject"
DB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
METRICS_PATH = os.path.join(PROJECT_DIR, "metrics.json")
ANCHORS_PATH = "/app/anchors.json"

RUN_ID = os.environ.get("ZEALT_RUN_ID", "")
TBL_O0 = f"chunks_o0_{RUN_ID}"
TBL_O75 = f"chunks_o75_{RUN_ID}"
TBL_O150 = f"chunks_o150_{RUN_ID}"


@pytest.fixture(scope="session")
def metrics():
    assert os.path.isfile(METRICS_PATH), f"Missing metrics file at {METRICS_PATH}."
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def db():
    assert os.path.isdir(DB_DIR), f"Missing LanceDB directory at {DB_DIR}."
    return lancedb.connect(DB_DIR)


@pytest.fixture(scope="session")
def anchors():
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 10, f"Fixture broken: expected 10 anchors, got {len(data)}."
    return data


def _check_float_in_unit_range(label, value):
    assert isinstance(value, float) or isinstance(value, int), (
        f"metrics.json: {label!r} value {value!r} is not a number."
    )
    fv = float(value)
    assert 0.0 <= fv <= 1.0, f"metrics.json: {label!r} value {fv} not in [0.0, 1.0]."


def test_run_id_set():
    assert RUN_ID, "ZEALT_RUN_ID must be set in the environment for verification."


def test_metrics_schema(metrics):
    assert set(metrics.keys()) == {"o0", "o75", "o150"}, (
        f"metrics.json top-level keys {list(metrics.keys())} != ['o0', 'o75', 'o150']."
    )
    for key in ("o0", "o75", "o150"):
        block = metrics[key]
        assert isinstance(block, dict), f"metrics[{key!r}] is not an object."
        assert set(block.keys()) == {"recall", "mrr"}, (
            f"metrics[{key!r}] keys {list(block.keys())} != ['recall', 'mrr']."
        )
        _check_float_in_unit_range(f"{key}.recall", block["recall"])
        _check_float_in_unit_range(f"{key}.mrr", block["mrr"])


def test_three_tables_exist(db):
    names = set(db.table_names())
    for tname in (TBL_O0, TBL_O75, TBL_O150):
        assert tname in names, (
            f"LanceDB table {tname!r} not found. Existing tables: {sorted(names)}"
        )


def _assert_schema(tbl):
    schema = tbl.schema
    fields = {f.name: f for f in schema}
    required = {"text", "doc_id", "start", "end", "vector"}
    missing = required - set(fields)
    assert not missing, (
        f"Table {tbl.name!r} missing required fields {missing}; has {list(fields)}."
    )
    assert pa.types.is_string(fields["text"].type), (
        f"Table {tbl.name!r} 'text' column must be string, got {fields['text'].type}."
    )
    assert pa.types.is_string(fields["doc_id"].type), (
        f"Table {tbl.name!r} 'doc_id' column must be string, got {fields['doc_id'].type}."
    )
    assert pa.types.is_integer(fields["start"].type), (
        f"Table {tbl.name!r} 'start' column must be integer, got {fields['start'].type}."
    )
    assert pa.types.is_integer(fields["end"].type), (
        f"Table {tbl.name!r} 'end' column must be integer, got {fields['end'].type}."
    )
    vec_type = fields["vector"].type
    assert pa.types.is_fixed_size_list(vec_type), (
        f"Table {tbl.name!r} 'vector' column must be fixed_size_list, got {vec_type}."
    )
    assert vec_type.list_size == 1536, (
        f"Table {tbl.name!r} vector dim {vec_type.list_size} != 1536 (text-embedding-3-small)."
    )


def test_schemas(db):
    for tname in (TBL_O0, TBL_O75, TBL_O150):
        _assert_schema(db.open_table(tname))


def test_row_counts_increase_with_overlap(db):
    n0 = db.open_table(TBL_O0).count_rows()
    n150 = db.open_table(TBL_O150).count_rows()
    assert n150 > n0, (
        f"Expected chunks_o150 row count > chunks_o0 row count, got n0={n0}, n150={n150}."
    )


def test_overlap_helps_recall(metrics):
    r0 = float(metrics["o0"]["recall"])
    r150 = float(metrics["o150"]["recall"])
    assert r150 > r0, (
        f"Expected recall(o150)={r150} > recall(o0)={r0}; overlap should help."
    )


def _spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def _recall_at_5_for_table(tbl, anchors, qvecs):
    hits = 0
    for anchor, qvec in zip(anchors, qvecs):
        rows = tbl.search(qvec).limit(5).to_list()
        for row in rows:
            if row.get("doc_id") != anchor["doc_id"]:
                continue
            if _spans_overlap(
                int(row["start"]), int(row["end"]),
                int(anchor["start"]), int(anchor["end"]),
            ):
                hits += 1
                break
    return hits / len(anchors)


def test_sanity_recompute_recall_o150(db, anchors, metrics):
    import openai

    client = openai.OpenAI()
    qvecs = []
    # Embed all anchor queries in one request to match candidate-side embeddings deterministically.
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[a["query"] for a in anchors],
    )
    qvecs = [item.embedding for item in resp.data]

    tbl = db.open_table(TBL_O150)
    independent_recall = _recall_at_5_for_table(tbl, anchors, qvecs)
    reported = float(metrics["o150"]["recall"])
    assert math.isclose(independent_recall, reported, abs_tol=1e-6), (
        f"Verifier-side Recall@5 for o150 = {independent_recall} but candidate reported {reported}."
    )
