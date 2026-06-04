import os
import numpy as np
import lancedb

uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("TABLE_NAME")
cat_filter = os.environ.get("CATEGORY_FILTER")
fts_query = os.environ.get("FTS_QUERY")

db = lancedb.connect(uri)
table = db.open_table(table_name)
qvec = np.load("query_vector.npy")

print("--- Plain ---")
plain_query = table.search(qvec).limit(10)
explain = plain_query.explain_plan()
print(explain)
analyze = plain_query.analyze_plan()
print(analyze)

print("--- Prefilter ---")
prefilter_query = table.search(qvec).where(f"category = '{cat_filter}'").limit(10)
print(prefilter_query.explain_plan())

print("--- Hybrid ---")
hybrid_query = table.search(query_type="hybrid").vector(qvec).text(fts_query).limit(10)
print(hybrid_query.explain_plan())
