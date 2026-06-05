import importlib
import json
import os
import subprocess
import sys

import pytest

PROJECT_DIR = "/home/user/myproject"
TRACE_LOG = "/tmp/otel_spans.jsonl"
LANCEDB_DIR = "/app/lancedb_data"


def _run_id():
    rid = os.environ.get("ZEALT_RUN_ID")
    assert rid, "ZEALT_RUN_ID environment variable must be set."
    return rid


def _read_spans(path):
    assert os.path.isfile(path), f"Trace log file {path} does not exist."
    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"Line {i} of {path} is not valid JSON: {e}; content={line!r}"
                )
            spans.append(obj)
    assert len(spans) > 0, f"Trace log file {path} is empty."
    return spans


def _check_required_keys(spans):
    required = {"name", "trace_id", "span_id", "parent_id", "attributes", "start_time", "end_time"}
    for sp in spans:
        missing = required - set(sp.keys())
        assert not missing, (
            f"Span {sp.get('name')!r} missing required keys {missing}; "
            f"available keys={list(sp.keys())}"
        )


@pytest.fixture(scope="module")
def cli_run():
    # Ensure a clean trace log before running the CLI.
    if os.path.isfile(TRACE_LOG):
        os.remove(TRACE_LOG)
    result = subprocess.run(
        ["python3", "run_queries.py", "--n", "6"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI failed (exit={result.returncode}). "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    spans = _read_spans(TRACE_LOG)
    return {"result": result, "spans": spans}


def test_cli_creates_trace_log(cli_run):
    assert os.path.isfile(TRACE_LOG), f"{TRACE_LOG} must exist after CLI run."


def test_spans_have_required_fields(cli_run):
    _check_required_keys(cli_run["spans"])


def test_connect_span_present(cli_run):
    names = [sp["name"] for sp in cli_run["spans"]]
    matching = [n for n in names if "connect" in n.lower()]
    assert matching, (
        "Expected at least one span name containing 'connect'; "
        f"got span names={names}"
    )


def _search_spans(spans):
    return [sp for sp in spans if "search" in sp["name"].lower()]


def test_search_spans_have_required_attributes(cli_run):
    spans = cli_run["spans"]
    rid = _run_id()
    expected_table = f"tracing_docs_{rid}"
    search_spans = _search_spans(spans)
    assert search_spans, "No 'search' spans found in trace log."

    for sp in search_spans:
        attrs = sp.get("attributes", {})
        assert attrs.get("lancedb.table") == expected_table, (
            f"Search span {sp['name']!r} has wrong lancedb.table attribute: "
            f"got {attrs.get('lancedb.table')!r}, expected {expected_table!r}"
        )
        qt = attrs.get("lancedb.query_type")
        assert qt in {"vector", "fts", "hybrid"}, (
            f"Search span has invalid lancedb.query_type={qt!r}; "
            "expected one of vector|fts|hybrid"
        )
        k = attrs.get("lancedb.k")
        assert isinstance(k, int) and k > 0, (
            f"Search span must have integer lancedb.k attribute; got {k!r}"
        )


def test_all_three_query_types_emitted(cli_run):
    spans = cli_run["spans"]
    qts = set()
    for sp in _search_spans(spans):
        qt = sp.get("attributes", {}).get("lancedb.query_type")
        if qt is not None:
            qts.add(qt)
    expected = {"vector", "fts", "hybrid"}
    assert expected <= qts, (
        f"Expected at least one search span per query type {expected}; "
        f"found query types={qts}"
    )


def _materialization_child_of(spans, search_span):
    children = []
    for sp in spans:
        if sp.get("trace_id") != search_span.get("trace_id"):
            continue
        if sp.get("parent_id") != search_span.get("span_id"):
            continue
        children.append(sp)
    return children


def test_materialization_children_carry_metrics(cli_run):
    spans = cli_run["spans"]
    search_spans = _search_spans(spans)
    assert search_spans, "No search spans to evaluate."

    for ss in search_spans:
        children = _materialization_child_of(spans, ss)
        # Find at least one child carrying the materialization metrics.
        good = []
        for ch in children:
            attrs = ch.get("attributes", {})
            rc = attrs.get("lancedb.result_count")
            lat = attrs.get("lancedb.latency_ms")
            if (
                isinstance(rc, int)
                and isinstance(lat, (int, float))
                and lat >= 0
            ):
                good.append(ch)
        assert good, (
            f"Search span name={ss['name']!r} span_id={ss['span_id']!r} "
            f"has no child span with lancedb.result_count + lancedb.latency_ms. "
            f"Direct children: "
            f"{[{ 'name': c['name'], 'attrs': c.get('attributes') } for c in children]}"
        )


def test_parent_child_trace_invariant_holds(cli_run):
    spans = cli_run["spans"]
    search_ids = {sp["span_id"] for sp in _search_spans(spans)}
    # Every materialization span must point at a real parent span_id in same trace.
    by_trace = {}
    for sp in spans:
        by_trace.setdefault(sp["trace_id"], {})[sp["span_id"]] = sp

    found_any_child = False
    for sp in spans:
        attrs = sp.get("attributes", {})
        if (
            isinstance(attrs.get("lancedb.result_count"), int)
            and isinstance(attrs.get("lancedb.latency_ms"), (int, float))
        ):
            pid = sp.get("parent_id")
            assert pid, (
                f"Materialization span {sp['name']!r} must have a non-null parent_id."
            )
            same_trace = by_trace.get(sp["trace_id"], {})
            assert pid in same_trace, (
                f"Materialization span {sp['name']!r} parent_id={pid!r} "
                f"not found among spans of trace {sp['trace_id']!r}."
            )
            assert pid in search_ids, (
                f"Materialization span {sp['name']!r} parent_id={pid!r} "
                "does not match any search span's span_id."
            )
            found_any_child = True
    assert found_any_child, (
        "Did not find any materialization child span carrying both "
        "lancedb.result_count and lancedb.latency_ms attributes."
    )


def test_service_search_returns_lists(cli_run):
    """Re-import solution and exercise vector + fts searches directly."""
    sys.path.insert(0, PROJECT_DIR)
    try:
        if "solution" in sys.modules:
            del sys.modules["solution"]
        mod = importlib.import_module("solution")
    finally:
        sys.path.pop(0)

    Service = getattr(mod, "LanceDBSearchService", None)
    assert Service is not None, "solution.py must export LanceDBSearchService."

    svc = Service()

    out = svc.search([0.0] * 32, k=3, query_type="vector")
    assert isinstance(out, list), f"vector search must return a list; got {type(out)!r}"
    assert len(out) <= 3, f"vector search returned more than k=3 rows: {len(out)}"
    for row in out:
        assert isinstance(row, dict), f"vector search rows must be dicts; got {type(row)!r}"

    out = svc.search("alpha", k=2, query_type="fts")
    assert isinstance(out, list), f"fts search must return a list; got {type(out)!r}"
    assert len(out) <= 2, f"fts search returned more than k=2 rows: {len(out)}"
    for row in out:
        assert isinstance(row, dict), f"fts search rows must be dicts; got {type(row)!r}"

    # Force-flush spans by shutting the provider down if available.
    shutdown_fn = getattr(mod, "shutdown_tracing", None)
    if callable(shutdown_fn):
        shutdown_fn()

    spans_after = _read_spans(TRACE_LOG)
    # We expect to find more search spans now, and the invariant from earlier still holds.
    assert len(spans_after) >= len(cli_run["spans"]), (
        "Expected the trace log to grow after running additional searches; "
        f"before={len(cli_run['spans'])} after={len(spans_after)}"
    )
