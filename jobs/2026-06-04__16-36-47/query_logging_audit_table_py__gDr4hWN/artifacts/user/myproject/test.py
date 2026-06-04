import lancedb
import pyarrow as pa
import numpy as np
from solution import LoggedSearcher

db = lancedb.connect("data.lancedb")
if "articles" in db.table_names():
    db.drop_table("articles")
db.create_table("articles", data=[{"id": 1, "title": "A", "embedding": np.random.rand(64).astype(np.float32)}])

searcher = LoggedSearcher("data.lancedb", "articles", "query_logs")
hits = searcher.search(np.random.rand(64).astype(np.float32), 1, "q1", "u1", "test")
print(hits)

logs = db.open_table("query_logs").to_pandas()
print(logs)
