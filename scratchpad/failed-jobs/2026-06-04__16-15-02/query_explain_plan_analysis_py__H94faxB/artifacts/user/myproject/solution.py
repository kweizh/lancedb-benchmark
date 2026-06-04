import os
import json
import re

import lancedb
import numpy as np


def parse_operators(explain_plan_text: str) -> list[str]:
    """Extract operator class names from the explain plan tree.

    Each non-empty line that contains a ':' in its leading token
    (before the first whitespace) is considered an operator node.
    """
    operators = []
    for line in explain_plan_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # The operator name is the first colon-delimited token on the line,
        # but only if the colon appears before any space (i.e. it's part of
        # the operator identifier like "ProjectionExec:" or "Take:").
        match = re.match(r'^([A-Za-z_]\w*):', stripped)
        if match:
            operators.append(match.group(1))
    return operators


def extract_top_output_rows(analyze_plan_text: str) -> int:
    """Extract output_rows from the root operator's metrics in the analyze plan.

    The root operator (first non-empty line after 'AnalyzeExec') contains
    the top-level output_rows metric.
    """
    # Find the first occurrence of output_rows=N in the analyze plan
    match = re.search(r'output_rows=(\d+)', analyze_plan_text)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not find output_rows in analyze plan:\n{analyze_plan_text}")


def main():
    # Read configuration from environment variables
    lancedb_uri = os.environ["LANCEDB_URI"]
    table_name = os.environ["TABLE_NAME"]
    category_filter = os.environ["CATEGORY_FILTER"]
    fts_query = os.environ["FTS_QUERY"]

    # Connect to the database and open the table
    db = lancedb.connect(lancedb_uri)
    table = db.open_table(table_name)

    # Load the query vector
    qvec = np.load("/home/user/myproject/query_vector.npy")

    report = {}

    # --- 1. Plain vector search ---
    plain_query = table.search(qvec).limit(10)
    plain_explain = plain_query.explain_plan()
    plain_analyze = plain_query.analyze_plan()
    report["plain"] = {
        "explain_plan": plain_explain,
        "analyze_plan": plain_analyze,
        "operators": parse_operators(plain_explain),
        "top_output_rows": extract_top_output_rows(plain_analyze),
    }

    # --- 2. Prefiltered vector search ---
    prefilter_query = (
        table.search(qvec)
        .where(f"category = '{category_filter}'")
        .limit(10)
    )
    prefilter_explain = prefilter_query.explain_plan()
    prefilter_analyze = prefilter_query.analyze_plan()
    report["prefilter"] = {
        "explain_plan": prefilter_explain,
        "analyze_plan": prefilter_analyze,
        "operators": parse_operators(prefilter_explain),
        "top_output_rows": extract_top_output_rows(prefilter_analyze),
    }

    # --- 3. Hybrid (vector + FTS) search ---
    hybrid_query = (
        table.search(query_type="hybrid")
        .vector(qvec)
        .text(fts_query)
        .limit(10)
    )
    hybrid_explain = hybrid_query.explain_plan()
    hybrid_analyze = hybrid_query.analyze_plan()
    report["hybrid"] = {
        "explain_plan": hybrid_explain,
        "analyze_plan": hybrid_analyze,
        "operators": parse_operators(hybrid_explain),
        "top_output_rows": extract_top_output_rows(hybrid_analyze),
    }

    # Write the report
    with open("/home/user/myproject/report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Report written to /home/user/myproject/report.json")


if __name__ == "__main__":
    main()