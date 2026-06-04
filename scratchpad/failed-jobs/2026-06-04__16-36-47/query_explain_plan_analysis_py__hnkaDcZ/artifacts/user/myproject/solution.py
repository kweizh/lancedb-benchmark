import os
import json
import re
import numpy as np
import lancedb

def parse_operators(explain_plan_str):
    ops = []
    for line in explain_plan_str.splitlines():
        if not line.strip():
            continue
        m = re.match(r'^\s*([A-Za-z0-9_]+):', line)
        if m:
            ops.append(m.group(1))
    return ops

def extract_top_output_rows(analyze_plan_str):
    m = re.search(r'output_rows=(\d+)', analyze_plan_str)
    if m:
        return int(m.group(1))
    return None

def main():
    uri = os.environ.get("LANCEDB_URI")
    table_name = os.environ.get("TABLE_NAME")
    cat_filter = os.environ.get("CATEGORY_FILTER")
    fts_query = os.environ.get("FTS_QUERY")

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    qvec = np.load("query_vector.npy")

    report = {}

    # 1. Plain vector search
    plain_query = table.search(qvec).limit(10)
    plain_explain = plain_query.explain_plan()
    plain_analyze = plain_query.analyze_plan()
    report["plain"] = {
        "explain_plan": plain_explain,
        "analyze_plan": plain_analyze,
        "operators": parse_operators(plain_explain),
        "top_output_rows": extract_top_output_rows(plain_analyze)
    }

    # 2. Prefiltered vector search
    prefilter_query = table.search(qvec).where(f"category = '{cat_filter}'").limit(10)
    prefilter_explain = prefilter_query.explain_plan()
    prefilter_analyze = prefilter_query.analyze_plan()
    report["prefilter"] = {
        "explain_plan": prefilter_explain,
        "analyze_plan": prefilter_analyze,
        "operators": parse_operators(prefilter_explain),
        "top_output_rows": extract_top_output_rows(prefilter_analyze)
    }

    # 3. Hybrid (vector + FTS) search
    hybrid_query = table.search(query_type="hybrid").vector(qvec).text(fts_query).limit(10)
    hybrid_explain = hybrid_query.explain_plan()
    hybrid_analyze = hybrid_query.analyze_plan()
    report["hybrid"] = {
        "explain_plan": hybrid_explain,
        "analyze_plan": hybrid_analyze,
        "operators": parse_operators(hybrid_explain),
        "top_output_rows": extract_top_output_rows(hybrid_analyze)
    }

    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
