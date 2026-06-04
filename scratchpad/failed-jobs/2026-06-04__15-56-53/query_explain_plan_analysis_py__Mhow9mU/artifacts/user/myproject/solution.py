"""
Query Plan Introspection with LanceDB
Produces a JSON report with physical plan details for three query types.
"""

import json
import os
import re

import lancedb
import numpy as np


def parse_operators(explain_text: str) -> list[str]:
    """Parse operator class names from explain plan text.

    Extracts the leading token before ':' on each non-empty line.
    For hybrid plans that contain section headers like 'Vector Search Plan:'
    or 'FTS Search Plan:', those lines are also captured as operator-like tokens.
    Only tokens that look like operator names (start with uppercase, no spaces) are kept.
    """
    operators = []
    for line in explain_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Match everything up to the first colon (operator name / header token)
        m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)\s*:', stripped)
        if m:
            operators.append(m.group(1))
    return operators


def extract_top_output_rows(analyze_text: str) -> int:
    """Extract output_rows from the first metrics block in the analyze plan.

    The first operator with metrics (after AnalyzeExec / TracedExec preamble)
    represents the root of the execution plan and therefore the final row count.
    """
    # Find the first occurrence of output_rows=<N> in a metrics block
    m = re.search(r'output_rows=(\d+)', analyze_text)
    if m:
        return int(m.group(1))
    return -1


def build_query_section(explain_text: str, analyze_text: str) -> dict:
    return {
        "explain_plan": explain_text,
        "analyze_plan": analyze_text,
        "operators": parse_operators(explain_text),
        "top_output_rows": extract_top_output_rows(analyze_text),
    }


def main():
    # ── Environment variables ────────────────────────────────────────────────
    db_uri = os.environ["LANCEDB_URI"]
    table_name = os.environ["TABLE_NAME"]
    category_filter = os.environ["CATEGORY_FILTER"]
    fts_query = os.environ["FTS_QUERY"]

    # ── Load assets ──────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qvec = np.load(os.path.join(script_dir, "query_vector.npy"))

    db = lancedb.connect(db_uri)
    tbl = db.open_table(table_name)

    # ── 1. Plain vector search ───────────────────────────────────────────────
    plain_q = tbl.search(qvec).limit(10)
    plain_explain = plain_q.explain_plan(verbose=True)
    plain_analyze = plain_q.analyze_plan()
    plain_section = build_query_section(plain_explain, plain_analyze)

    # ── 2. Prefiltered vector search ─────────────────────────────────────────
    prefilter_q = (
        tbl.search(qvec)
        .where(f"category = '{category_filter}'", prefilter=True)
        .limit(10)
    )
    prefilter_explain = prefilter_q.explain_plan(verbose=True)
    prefilter_analyze = prefilter_q.analyze_plan()
    prefilter_section = build_query_section(prefilter_explain, prefilter_analyze)

    # ── 3. Hybrid (vector + FTS) search ─────────────────────────────────────
    hybrid_q = (
        tbl.search(query_type="hybrid")
        .vector(qvec)
        .text(fts_query)
        .limit(10)
    )
    hybrid_explain = hybrid_q.explain_plan(verbose=True)
    hybrid_analyze = hybrid_q.analyze_plan()
    hybrid_section = build_query_section(hybrid_explain, hybrid_analyze)

    # ── Assemble & write report ──────────────────────────────────────────────
    report = {
        "plain": plain_section,
        "prefilter": prefilter_section,
        "hybrid": hybrid_section,
    }

    report_path = os.path.join(script_dir, "report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"Report written to {report_path}")

    # ── Quick sanity check ───────────────────────────────────────────────────
    assert "plain" in report and "prefilter" in report and "hybrid" in report
    assert report["plain"]["top_output_rows"] == 10
    assert report["prefilter"]["top_output_rows"] == 10
    assert any(op.startswith("Filter") for op in report["prefilter"]["operators"]), \
        "prefilter plan missing Filter operator"
    assert any("Projection" in op for op in report["prefilter"]["operators"]), \
        "prefilter plan missing Projection operator"
    assert any("Take" in op for op in report["prefilter"]["operators"]), \
        "prefilter plan missing Take operator"
    hybrid_plan_lower = report["hybrid"]["explain_plan"].lower()
    assert "vector" in hybrid_plan_lower, "hybrid plan missing Vector sub-plan"
    assert "fts" in hybrid_plan_lower, "hybrid plan missing FTS sub-plan"
    assert len(report["hybrid"]["operators"]) > 0, "hybrid operators list is empty"

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
