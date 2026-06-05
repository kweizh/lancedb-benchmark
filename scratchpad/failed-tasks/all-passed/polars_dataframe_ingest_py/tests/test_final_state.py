import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest


PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
LOG_PATH = os.path.join(PROJECT_DIR, "output.log")
DB_URI = os.path.join(PROJECT_DIR, "lancedb")


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID must be set in the verifier environment."
    return rid


def _table_name() -> str:
    return f"polars_ingest_{_run_id()}"


def _polars_source():
    """Rebuild the deterministic source identical to the candidate's build_dataframe."""
    rng = np.random.default_rng(2026)
    ids = np.arange(500, dtype=np.int64)
    titles = [f"item-{i}" for i in range(500)]
    scores = rng.uniform(0.0, 1.0, size=500)
    tags = rng.choice(["alpha", "beta", "gamma", "delta"], size=500)
    vectors = rng.standard_normal((500, 32)).astype(np.float32)
    return ids, titles, scores, tags, vectors


@pytest.fixture(scope="session")
def run_solution():
    # Clean previous artifacts so we measure a fresh run.
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    if os.path.isdir(DB_URI):
        import shutil
        shutil.rmtree(DB_URI)

    assert os.path.isfile(SOLUTION_PATH), f"{SOLUTION_PATH} must exist (candidate solution)."

    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "solution.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    # LanceDB 0.25.3 can SIGABRT (exit -6) during Python interpreter teardown
    # AFTER the table has been fully written and the JSON line has been flushed.
    # Tolerate that specific case so the rest of the verifier can inspect outcomes;
    # any other non-zero exit still fails.
    assert proc.returncode in (0, -6), (
        f"`python3 solution.py` failed (exit={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # Sanity-check that the script actually produced stdout before the (potential) abort.
    assert proc.stdout.strip(), (
        "`python3 solution.py` produced no stdout."
        f" STDERR:\n{proc.stderr}"
    )
    return proc


@pytest.fixture(scope="session")
def lance_table(run_solution):
    import lancedb

    db = lancedb.connect(DB_URI)
    names = db.table_names()
    tname = _table_name()
    assert tname in names, f"Expected LanceDB table {tname!r}, found: {names}"
    return db.open_table(tname)


def test_stdout_is_single_json_line(run_solution):
    out = run_solution.stdout.strip()
    assert out, "solution.py must print a JSON line to stdout."
    # The single JSON line must be parseable as a list of objects.
    last_line = [ln for ln in out.splitlines() if ln.strip()][-1]
    parsed = json.loads(last_line)
    assert isinstance(parsed, list), f"Expected stdout JSON to be a list, got {type(parsed).__name__}"
    if parsed:
        assert isinstance(parsed[0], dict), "Each search result must be a dict."


def test_log_file_contains_demo_json(run_solution):
    assert os.path.isfile(LOG_PATH), f"{LOG_PATH} must be written by solution.py."
    with open(LOG_PATH, "r") as f:
        text = f.read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines, "output.log must contain at least one non-empty line."
    # The last non-empty line must be valid JSON list.
    parsed = json.loads(lines[-1])
    assert isinstance(parsed, list), "The final non-empty log line must be a JSON array."


def test_table_row_count(lance_table):
    assert lance_table.count_rows() == 500, (
        f"Table {_table_name()} must contain exactly 500 rows, got {lance_table.count_rows()}."
    )


def test_table_schema_preserved(lance_table):
    schema = lance_table.schema
    field_names = [f.name for f in schema]
    assert field_names == ["id", "title", "score", "tag", "vector"], (
        f"Column order must be ['id', 'title', 'score', 'tag', 'vector'], got {field_names}"
    )

    id_type = schema.field("id").type
    score_type = schema.field("score").type
    title_type = schema.field("title").type
    tag_type = schema.field("tag").type
    vec_type = schema.field("vector").type

    assert pa.types.is_int64(id_type), f"id must be int64, got {id_type}"
    assert pa.types.is_floating(score_type) and score_type.bit_width == 64, (
        f"score must be float64, got {score_type}"
    )
    assert pa.types.is_string(title_type) or pa.types.is_large_string(title_type), (
        f"title must be string/large_string, got {title_type}"
    )
    assert pa.types.is_string(tag_type) or pa.types.is_large_string(tag_type), (
        f"tag must be string/large_string, got {tag_type}"
    )

    # Vector must be a list of float32 with width 32.
    assert pa.types.is_list(vec_type) or pa.types.is_fixed_size_list(vec_type) or pa.types.is_large_list(vec_type), (
        f"vector must be a list-of-float32 type, got {vec_type}"
    )
    inner = vec_type.value_type
    assert pa.types.is_float32(inner), f"vector inner type must be float32, got {inner}"

    # Spot-check that every row has 32 elements in the vector column.
    sample = lance_table.to_pandas()
    assert len(sample) == 500
    sample_vectors = sample["vector"].tolist()
    assert all(len(v) == 32 for v in sample_vectors), (
        "Every row's vector must contain exactly 32 float32 values."
    )


def test_table_values_match_polars_source(lance_table):
    ids_src, titles_src, scores_src, tags_src, _ = _polars_source()
    df = lance_table.to_pandas().sort_values("id").reset_index(drop=True)

    assert df["id"].tolist() == ids_src.tolist(), "id column must equal 0..499 from polars source."
    assert df["title"].tolist() == titles_src, "title column must equal 'item-<id>' values."
    assert df["tag"].tolist() == tags_src.tolist(), (
        "tag column values must match polars source row-by-row after sorting by id."
    )
    np.testing.assert_allclose(
        df["score"].to_numpy(dtype=np.float64),
        scores_src,
        atol=1e-6,
        err_msg="score column must match polars source within 1e-6 tolerance.",
    )


def test_search_alpha_min_score_05(lance_table):
    sys.path.insert(0, PROJECT_DIR)
    import importlib
    import solution  # type: ignore

    importlib.reload(solution)
    vec = [0.0] * 32
    results = solution.search(lance_table, vec, top_k=10, min_score=0.5, tag="alpha")
    assert isinstance(results, list), "search() must return a list."
    assert results, (
        "With the deterministic seed there are 59 alpha rows with score>=0.5; "
        "search() with top_k=10 must return a non-empty list."
    )
    assert len(results) <= 10, f"search() returned {len(results)} > top_k=10."

    distances = []
    for row in results:
        for key in ("id", "title", "score", "tag", "_distance"):
            assert key in row, f"Result row is missing key {key!r}: {row}"
        assert row["tag"] == "alpha", f"Row violated tag filter: {row}"
        assert float(row["score"]) >= 0.5, f"Row violated score filter: {row}"
        distances.append(float(row["_distance"]))

    assert distances == sorted(distances), (
        f"_distance values must be non-decreasing, got {distances}"
    )


def test_search_beta_min_score_09(lance_table):
    sys.path.insert(0, PROJECT_DIR)
    import importlib
    import solution  # type: ignore

    importlib.reload(solution)
    vec = [0.0] * 32
    results = solution.search(lance_table, vec, top_k=5, min_score=0.9, tag="beta")
    assert isinstance(results, list), "search() must return a list."
    assert len(results) <= 5, f"search() returned {len(results)} > top_k=5."
    # With the deterministic seed there are exactly 9 beta rows with score >= 0.9.
    assert results, "Expected at least one match for tag=beta and score>=0.9."

    distances = []
    for row in results:
        assert row["tag"] == "beta", f"Row violated tag filter: {row}"
        assert float(row["score"]) >= 0.9, f"Row violated score filter: {row}"
        distances.append(float(row["_distance"]))

    assert distances == sorted(distances), (
        f"_distance values must be non-decreasing, got {distances}"
    )
