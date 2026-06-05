import importlib
import json
import os
import re
import subprocess
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")

REQUIRED_FIELDS = {
    "src_id": "int64",
    "dst_id": "int64",
    "rank": "int32",
    "distance": "float",
}


def _run_id() -> str:
    run_id = os.environ.get("ZEALT_RUN_ID")
    assert run_id, "ZEALT_RUN_ID must be set in the verifier environment."
    return run_id


def _edges_name() -> str:
    return f"knn_edges_{_run_id()}"


def _embeddings_name() -> str:
    return f"embeddings_{_run_id()}"


@pytest.fixture(scope="session")
def build_output():
    """Drop any pre-existing knn_edges table, then run build_graph.py once and capture stdout."""
    import lancedb  # noqa: F401  (ensure import works)

    db = lancedb.connect(LANCEDB_DIR)
    edges_name = _edges_name()
    if edges_name in db.table_names():
        db.drop_table(edges_name)

    proc = subprocess.run(
        ["python3", "build_graph.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"build_graph.py exited with {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return proc


@pytest.fixture(scope="session")
def edges_arrow(build_output):
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    edges_name = _edges_name()
    assert edges_name in db.table_names(), (
        f"Edges table {edges_name} was not created by build_graph.py. "
        f"build stdout: {build_output.stdout}"
    )
    tbl = db.open_table(edges_name)
    return tbl.to_arrow()


@pytest.fixture(scope="session")
def source_ids():
    import lancedb

    db = lancedb.connect(LANCEDB_DIR)
    tbl = db.open_table(_embeddings_name())
    return sorted(tbl.to_pandas()["id"].astype(int).tolist())


@pytest.fixture(scope="session")
def reference_graph(edges_arrow):
    import networkx as nx

    df = edges_arrow.to_pandas()
    g = nx.DiGraph()
    for _, row in df.iterrows():
        src = int(row["src_id"])
        dst = int(row["dst_id"])
        if src == dst:
            continue
        g.add_edge(src, dst)
    return g


@pytest.fixture(scope="session")
def query_pairs(source_ids):
    import numpy as np

    rng = np.random.default_rng(2026)
    ids = list(source_ids)
    pairs = []
    seen = set()
    while len(pairs) < 8:
        a, b = rng.choice(ids, size=2, replace=False)
        a, b = int(a), int(b)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def test_build_stdout_row_count(build_output):
    pattern = re.compile(r"Built knn_edges rows=5000")
    assert pattern.search(build_output.stdout), (
        f"Expected stdout to contain 'Built knn_edges rows=5000'.\n"
        f"STDOUT:\n{build_output.stdout}\nSTDERR:\n{build_output.stderr}"
    )


def test_edges_table_row_count(edges_arrow):
    assert edges_arrow.num_rows == 5000, (
        f"Expected exactly 5000 rows in {_edges_name()}, got {edges_arrow.num_rows}."
    )


def test_edges_table_schema(edges_arrow):
    import pyarrow as pa

    schema = edges_arrow.schema
    field_map = {f.name: f.type for f in schema}
    missing = set(REQUIRED_FIELDS) - set(field_map)
    assert not missing, f"Missing required fields in edges schema: {missing}"
    assert pa.types.is_int64(field_map["src_id"]), (
        f"src_id must be int64, got {field_map['src_id']}"
    )
    assert pa.types.is_int64(field_map["dst_id"]), (
        f"dst_id must be int64, got {field_map['dst_id']}"
    )
    assert pa.types.is_int32(field_map["rank"]), (
        f"rank must be int32, got {field_map['rank']}"
    )
    assert pa.types.is_float32(field_map["distance"]), (
        f"distance must be float32, got {field_map['distance']}"
    )


def test_edges_per_source_structure(edges_arrow, source_ids):
    df = edges_arrow.to_pandas()
    grouped = df.groupby("src_id")
    seen_sources = set(int(s) for s in grouped.groups.keys())
    expected_sources = set(source_ids)
    assert seen_sources == expected_sources, (
        f"Edges table src_id set mismatch. "
        f"Missing: {sorted(expected_sources - seen_sources)[:10]}... "
        f"Extra: {sorted(seen_sources - expected_sources)[:10]}..."
    )

    expected_ranks = list(range(10))
    for src_id, group in grouped:
        sorted_group = group.sort_values("rank").reset_index(drop=True)
        ranks = sorted_group["rank"].astype(int).tolist()
        assert ranks == expected_ranks, (
            f"src_id={int(src_id)}: rank values must be [0..9] sorted, got {ranks}"
        )
        distances = sorted_group["distance"].astype(float).tolist()
        for i in range(1, len(distances)):
            assert distances[i] >= distances[i - 1] - 1e-6, (
                f"src_id={int(src_id)}: distances not monotonic at rank {i}: "
                f"{distances[i - 1]} -> {distances[i]}"
            )


def test_no_self_loops_in_nonzero_rank(edges_arrow):
    df = edges_arrow.to_pandas()
    bad = df[(df["rank"] >= 1) & (df["src_id"] == df["dst_id"])]
    assert len(bad) == 0, (
        f"Found {len(bad)} self-loops at rank >= 1; rank-0 may be self-match but ranks 1..9 must not be."
    )


def _validate_path(path, src, dst, graph, max_hops):
    assert isinstance(path, list) and len(path) >= 2, (
        f"path must be a list of node ids of length >= 2, got {path!r}"
    )
    assert int(path[0]) == src, f"path must start with {src}, got {path[0]}"
    assert int(path[-1]) == dst, f"path must end with {dst}, got {path[-1]}"
    assert len(path) - 1 <= max_hops, (
        f"path has {len(path) - 1} hops, exceeds max_hops={max_hops}"
    )
    for u, v in zip(path[:-1], path[1:]):
        assert graph.has_edge(int(u), int(v)), (
            f"path edge ({u}, {v}) is not present in the verifier reference graph"
        )


def _nx_path_within(graph, src, dst, max_hops):
    import networkx as nx

    try:
        sp = nx.shortest_path(graph, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(sp) - 1 > max_hops:
        return None
    return sp


def test_find_path_matches_networkx(build_output, reference_graph, query_pairs):
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    # Force a fresh import in case a prior test imported a stale module.
    if "solution" in sys.modules:
        del sys.modules["solution"]
    solution = importlib.import_module("solution")
    assert hasattr(solution, "find_path"), "solution.py must expose find_path."

    max_hops = 4
    for src, dst in query_pairs:
        expected = _nx_path_within(reference_graph, src, dst, max_hops)
        result = solution.find_path(src, dst, max_hops)
        if expected is None:
            assert result is None, (
                f"find_path({src},{dst},{max_hops}) returned {result!r} "
                f"but networkx finds no path within {max_hops} hops."
            )
        else:
            assert result is not None, (
                f"find_path({src},{dst},{max_hops}) returned None "
                f"but networkx finds a path {expected} within {max_hops} hops."
            )
            _validate_path(result, src, dst, reference_graph, max_hops)


def test_query_path_cli(build_output, reference_graph, query_pairs):
    # Pick the first pair where networkx finds a path within 4 hops; otherwise fall back to the first pair.
    chosen = None
    for src, dst in query_pairs:
        if _nx_path_within(reference_graph, src, dst, 4) is not None:
            chosen = (src, dst)
            break
    if chosen is None:
        chosen = query_pairs[0]
    src, dst = chosen

    proc = subprocess.run(
        [
            "python3",
            "query_path.py",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--max-hops",
            "4",
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"query_path.py failed (returncode {proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # Find the JSON line in stdout.
    payload = None
    for line in proc.stdout.strip().splitlines()[::-1]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "path" in obj:
            payload = obj
            break
    assert payload is not None, (
        f"query_path.py must print a JSON line with key 'path'. STDOUT:\n{proc.stdout}"
    )

    expected = _nx_path_within(reference_graph, src, dst, 4)
    if expected is None:
        assert payload["path"] is None, (
            f"query_path.py returned {payload!r} but networkx finds no path within 4 hops."
        )
    else:
        assert payload["path"] is not None, (
            f"query_path.py returned null but networkx finds a path {expected} within 4 hops."
        )
        _validate_path(payload["path"], src, dst, reference_graph, 4)
