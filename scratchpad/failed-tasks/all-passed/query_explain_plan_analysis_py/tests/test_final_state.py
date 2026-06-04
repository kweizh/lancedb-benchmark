import json
import os
import re

import lancedb
import pytest

PROJECT_DIR = "/home/user/myproject"
REPORT_PATH = os.path.join(PROJECT_DIR, "report.json")

REQUIRED_KEYS = {"explain_plan", "analyze_plan", "operators", "top_output_rows"}


@pytest.fixture(scope="module")
def report():
    assert os.path.isfile(REPORT_PATH), f"Report file {REPORT_PATH} does not exist."
    with open(REPORT_PATH) as f:
        data = json.load(f)
    return data


def test_top_level_keys(report):
    assert set(report.keys()) == {"plain", "prefilter", "hybrid"}, (
        f"report.json must have exactly the keys plain/prefilter/hybrid; got {list(report.keys())}"
    )


@pytest.mark.parametrize("section", ["plain", "prefilter", "hybrid"])
def test_section_schema(report, section):
    sec = report[section]
    assert isinstance(sec, dict), f"Section {section} must be a JSON object."
    missing = REQUIRED_KEYS - set(sec.keys())
    assert not missing, f"Section {section} is missing keys: {missing}"
    assert isinstance(sec["explain_plan"], str) and sec["explain_plan"].strip(), (
        f"{section}.explain_plan must be a non-empty string."
    )
    assert isinstance(sec["analyze_plan"], str) and sec["analyze_plan"].strip(), (
        f"{section}.analyze_plan must be a non-empty string."
    )
    assert isinstance(sec["operators"], list) and all(
        isinstance(op, str) for op in sec["operators"]
    ), f"{section}.operators must be a list of strings."
    assert isinstance(sec["top_output_rows"], int), (
        f"{section}.top_output_rows must be an integer; got {type(sec['top_output_rows'])}"
    )


def test_plain_top_output_rows_is_ten(report):
    assert report["plain"]["top_output_rows"] == 10, (
        f"plain.top_output_rows must equal 10 (the requested limit); got {report['plain']['top_output_rows']}"
    )


def test_prefilter_top_output_rows_is_ten(report):
    assert report["prefilter"]["top_output_rows"] == 10, (
        f"prefilter.top_output_rows must equal 10 (the requested limit); got {report['prefilter']['top_output_rows']}"
    )


def test_prefilter_has_filter_projection_take(report):
    ops = report["prefilter"]["operators"]
    assert any(op.startswith("Filter") for op in ops), (
        f"prefilter.operators must contain a Filter* operator; got {ops}"
    )
    assert any("Projection" in op for op in ops), (
        f"prefilter.operators must contain a Projection operator; got {ops}"
    )
    assert any("Take" in op for op in ops), (
        f"prefilter.operators must contain a Take/RemoteTake operator; got {ops}"
    )


def test_plain_has_vector_search_operator(report):
    ops = report["plain"]["operators"]
    assert any(("KNN" in op) or ("Vector" in op) or ("ANN" in op) for op in ops), (
        f"plain.operators must contain a vector search operator (KNN*/Vector*/ANN*); got {ops}"
    )


def test_hybrid_explain_plan_mentions_both_subplans(report):
    text = report["hybrid"]["explain_plan"].lower()
    assert "vector" in text, "hybrid.explain_plan must reference the vector sub-plan."
    assert "fts" in text, "hybrid.explain_plan must reference the FTS sub-plan."


def test_hybrid_operators_non_empty(report):
    ops = report["hybrid"]["operators"]
    assert isinstance(ops, list) and len(ops) > 0, (
        f"hybrid.operators must be a non-empty list; got {ops}"
    )


def test_analyze_plan_contains_metrics(report):
    metric_re = re.compile(r"metrics=\[[^\]]*output_rows=\d+", re.IGNORECASE)
    for section in ("plain", "prefilter", "hybrid"):
        assert metric_re.search(report[section]["analyze_plan"]), (
            f"{section}.analyze_plan must include metrics=[... output_rows=N ...] blocks."
        )


def test_table_still_intact():
    uri = os.environ["LANCEDB_URI"]
    name = os.environ["TABLE_NAME"]
    db = lancedb.connect(uri)
    assert name in db.table_names(), f"Table {name} disappeared from database."
    tbl = db.open_table(name)
    assert tbl.count_rows() >= 256, (
        f"Table {name} must still contain at least 256 rows after evaluation."
    )
    indices = tbl.list_indices()
    fts_cols = [list(getattr(i, "columns", [])) for i in indices]
    assert any(cols == ["text"] for cols in fts_cols), (
        f"Expected FTS index on text to remain after evaluation; got {indices}"
    )
