import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = "/home/user/myproject"
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution.py")
REPORT_PATH = os.path.join(PROJECT_DIR, "size_report.json")
DATA_DIR = os.path.join(PROJECT_DIR, "lancedb_data")


def _run_id() -> str:
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID environment variable is not set."
    return rid


def _table_dir(run_id: str, kind: str) -> str:
    # LanceDB stores each table as <name>.lance/ under the connect dir.
    return os.path.join(DATA_DIR, f"{kind}_{run_id}.lance")


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


@pytest.fixture(scope="module")
def run_solution():
    # Clean previous artifacts.
    if os.path.isdir(DATA_DIR):
        import shutil

        shutil.rmtree(DATA_DIR)
    if os.path.isfile(REPORT_PATH):
        os.remove(REPORT_PATH)

    assert os.path.isfile(SOLUTION_PATH), f"Missing solution.py at {SOLUTION_PATH}"
    result = subprocess.run(
        ["python3", SOLUTION_PATH],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"solution.py exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def test_stdout_format(run_solution):
    pattern = re.compile(
        r"default_bytes=(\d+)\s+zstd_bytes=(\d+)\s+ratio=([0-9]*\.?[0-9]+)"
    )
    assert pattern.search(run_solution.stdout), (
        f"Expected stdout to contain a line like "
        f"'default_bytes=<int> zstd_bytes=<int> ratio=<float>'. "
        f"Got stdout:\n{run_solution.stdout}"
    )


def test_size_report_exists_and_parses(run_solution):
    assert os.path.isfile(REPORT_PATH), f"Missing size_report.json at {REPORT_PATH}"
    with open(REPORT_PATH) as f:
        report = json.load(f)
    assert isinstance(report, dict), "size_report.json must be a JSON object."
    assert "default_bytes" in report, "size_report.json missing 'default_bytes'."
    assert "zstd_bytes" in report, "size_report.json missing 'zstd_bytes'."
    assert "ratio" in report, "size_report.json missing 'ratio'."
    assert isinstance(report["default_bytes"], int), (
        f"default_bytes must be int, got {type(report['default_bytes']).__name__}"
    )
    assert isinstance(report["zstd_bytes"], int), (
        f"zstd_bytes must be int, got {type(report['zstd_bytes']).__name__}"
    )
    assert isinstance(report["ratio"], (int, float)), (
        f"ratio must be numeric, got {type(report['ratio']).__name__}"
    )
    assert report["default_bytes"] > 0, "default_bytes should be positive."
    assert report["zstd_bytes"] > 0, "zstd_bytes should be positive."
    expected_ratio = report["zstd_bytes"] / report["default_bytes"]
    assert abs(report["ratio"] - expected_ratio) < 1e-9, (
        f"ratio in report ({report['ratio']}) does not match "
        f"zstd_bytes/default_bytes ({expected_ratio})."
    )


def test_both_tables_exist(run_solution):
    run_id = _run_id()
    import lancedb

    db = lancedb.connect(DATA_DIR)
    names = set(db.table_names())
    assert f"default_{run_id}" in names, (
        f"Expected table 'default_{run_id}' in {names}"
    )
    assert f"zstd_{run_id}" in names, f"Expected table 'zstd_{run_id}' in {names}"


def test_table_schemas_and_row_counts(run_solution):
    run_id = _run_id()
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(DATA_DIR)
    for kind in ("default", "zstd"):
        tbl = db.open_table(f"{kind}_{run_id}")
        assert tbl.count_rows() == 5000, (
            f"Table {kind}_{run_id} should have 5000 rows, got {tbl.count_rows()}"
        )
        schema = tbl.schema
        embedding_field = schema.field("embedding")
        # fixed_size_list<float32, 32>
        assert pa.types.is_fixed_size_list(embedding_field.type), (
            f"embedding column must be fixed_size_list, got {embedding_field.type}"
        )
        assert embedding_field.type.list_size == 32, (
            f"embedding fixed-size-list must have size 32, "
            f"got {embedding_field.type.list_size}"
        )
        assert pa.types.is_float32(embedding_field.type.value_type), (
            f"embedding inner type must be float32, "
            f"got {embedding_field.type.value_type}"
        )


def test_compression_is_effective(run_solution):
    with open(REPORT_PATH) as f:
        report = json.load(f)
    assert report["zstd_bytes"] < report["default_bytes"] * 0.95, (
        f"zstd table size {report['zstd_bytes']} is not at least 5% smaller than "
        f"default table size {report['default_bytes']} "
        f"(ratio={report['ratio']:.4f})."
    )
    assert 0 < report["ratio"] <= 0.95, (
        f"ratio must be in (0, 0.95], got {report['ratio']}"
    )


def test_compare_sizes_function(run_solution):
    spec = importlib.util.spec_from_file_location("solution_mod", SOLUTION_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution_mod"] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "compare_sizes"), (
        "solution.py must expose a top-level callable 'compare_sizes'."
    )
    result = mod.compare_sizes()
    assert isinstance(result, dict), "compare_sizes() must return a dict."
    for key in ("default_bytes", "zstd_bytes", "ratio"):
        assert key in result, f"compare_sizes() result missing key '{key}'"

    run_id = _run_id()
    default_dir = _table_dir(run_id, "default")
    zstd_dir = _table_dir(run_id, "zstd")
    assert os.path.isdir(default_dir), f"Missing default table dir: {default_dir}"
    assert os.path.isdir(zstd_dir), f"Missing zstd table dir: {zstd_dir}"
    expected_default = _dir_bytes(default_dir)
    expected_zstd = _dir_bytes(zstd_dir)
    assert result["default_bytes"] == expected_default, (
        f"compare_sizes() default_bytes {result['default_bytes']} != "
        f"on-disk sum {expected_default} for {default_dir}"
    )
    assert result["zstd_bytes"] == expected_zstd, (
        f"compare_sizes() zstd_bytes {result['zstd_bytes']} != "
        f"on-disk sum {expected_zstd} for {zstd_dir}"
    )

    with open(REPORT_PATH) as f:
        report = json.load(f)
    assert result["default_bytes"] == report["default_bytes"], (
        "compare_sizes() default_bytes disagrees with size_report.json."
    )
    assert result["zstd_bytes"] == report["zstd_bytes"], (
        "compare_sizes() zstd_bytes disagrees with size_report.json."
    )


def test_search_parity(run_solution):
    """Both tables must return identical top-5 ids for a fixed query vector."""
    run_id = _run_id()
    import lancedb

    db = lancedb.connect(DATA_DIR)
    default_tbl = db.open_table(f"default_{run_id}")
    zstd_tbl = db.open_table(f"zstd_{run_id}")

    qvec = np.random.default_rng(7).standard_normal(32).astype("float32")

    def top_ids(tbl):
        rows = tbl.search(qvec).limit(5).to_list()
        return [int(r["id"]) for r in rows]

    default_ids = top_ids(default_tbl)
    zstd_ids = top_ids(zstd_tbl)
    assert default_ids == zstd_ids, (
        f"Top-5 ids differ between tables. "
        f"default={default_ids} zstd={zstd_ids}"
    )
    assert len(default_ids) == 5, f"Expected 5 ids, got {default_ids}"
