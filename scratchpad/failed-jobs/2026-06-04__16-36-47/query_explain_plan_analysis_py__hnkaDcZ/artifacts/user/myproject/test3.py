import os
import numpy as np
import lancedb

uri = os.environ.get("LANCEDB_URI")
table_name = os.environ.get("TABLE_NAME")
fts_query = os.environ.get("FTS_QUERY")

db = lancedb.connect(uri)
table = db.open_table(table_name)
qvec = np.load("query_vector.npy")

hybrid_query = table.search(query_type="hybrid").vector(qvec).text(fts_query).limit(10)
print(hybrid_query.analyze_plan())
