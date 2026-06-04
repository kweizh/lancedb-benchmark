import json
import os
import re
import numpy as np
import lancedb

def parse_operators(explain_str):
    operators = []
    for line in explain_str.splitlines():
        if not line.strip():
            continue
        # Parse operator names with a regex on the leading token before ':'
        match = re.match(r'^\s*([a-zA-Z0-9_]+):', line)
        if match:
            operators.append(match.group(1))
    return operators

def extract_top_output_rows(analyze_str):
    match = re.search(r'output_rows=(\d+)', analyze_str)
    if match:
        return int(match.group(1))
    return None

def main():
    # 1. Retrieve environment variables
    lancedb_uri = os.environ.get('LANCEDB_URI', 'REDACTED_data')
    table_name = os.environ.get('TABLE_NAME', 'docs')
    category_filter = os.environ.get('CATEGORY_FILTER', 'alpha')
    fts_query = os.environ.get('FTS_QUERY', 'lancedb')

    # 2. Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    query_vector_path = os.path.join(base_dir, 'query_vector.npy')
    if not os.path.exists(query_vector_path):
        # Fallback to current working directory
        query_vector_path = 'query_vector.npy'

    report_path = os.path.join(base_dir, 'report.json')

    # 3. Load query vector
    qvec = np.load(query_vector_path)

    # 4. Open LanceDB table
    db = lancedb.connect(lancedb_uri)
    table = db.open_table(table_name)

    # 5. Plain vector search
    qb_plain = table.search(qvec).limit(10)
    plain_explain = qb_plain.explain_plan()
    plain_analyze = qb_plain.analyze_plan()
    plain_operators = parse_operators(plain_explain)
    plain_top_rows = extract_top_output_rows(plain_analyze)

    # 6. Prefiltered vector search
    # Filter must be pushed in as a prefilter
    qb_prefilter = table.search(qvec).where(f"category = '{category_filter}'", prefilter=True).limit(10)
    prefilter_explain = qb_prefilter.explain_plan()
    prefilter_analyze = qb_prefilter.analyze_plan()
    prefilter_operators = parse_operators(prefilter_explain)
    prefilter_top_rows = extract_top_output_rows(prefilter_analyze)

    # 7. Hybrid search
    qb_hybrid = table.search(query_type="hybrid").vector(qvec).text(fts_query).limit(10)
    hybrid_explain = qb_hybrid.explain_plan()
    hybrid_analyze = qb_hybrid.analyze_plan()
    hybrid_operators = parse_operators(hybrid_explain)
    hybrid_top_rows = extract_top_output_rows(hybrid_analyze)

    # 8. Structure the report
    report = {
        "plain": {
            "explain_plan": plain_explain,
            "analyze_plan": plain_analyze,
            "operators": plain_operators,
            "top_output_rows": plain_top_rows
        },
        "prefilter": {
            "explain_plan": prefilter_explain,
            "analyze_plan": prefilter_analyze,
            "operators": prefilter_operators,
            "top_output_rows": prefilter_top_rows
        },
        "hybrid": {
            "explain_plan": hybrid_explain,
            "analyze_plan": hybrid_analyze,
            "operators": hybrid_operators,
            "top_output_rows": hybrid_top_rows
        }
    }

    # 9. Write report.json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report successfully generated and written to {report_path}")

if __name__ == '__main__':
    main()
