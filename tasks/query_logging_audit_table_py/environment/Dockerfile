FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Seed the LanceDB database with the `articles` table (200 rows, 64-d vectors).
RUN python3 - <<'PYEOF'
import os
import numpy as np
import pyarrow as pa
import lancedb

DB_PATH = "/home/user/myproject/data.lancedb"
os.makedirs("/home/user/myproject", exist_ok=True)

rng = np.random.default_rng(seed=2026)
n_rows = 200
dim = 64
vecs = rng.standard_normal((n_rows, dim)).astype(np.float32)

ids = list(range(1, n_rows + 1))
titles = [f"Article #{i:03d}" for i in ids]

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("title", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), list_size=dim)),
])

table_data = pa.Table.from_pydict(
    {
        "id": ids,
        "title": titles,
        "embedding": [v.tolist() for v in vecs],
    },
    schema=schema,
)

db = lancedb.connect(DB_PATH)
if "articles" in db.table_names():
    db.drop_table("articles")
db.create_table("articles", data=table_data)
tbl = db.open_table("articles")
assert tbl.count_rows() == n_rows, f"seed failed: {tbl.count_rows()}"
print("articles table seeded:", tbl.count_rows())

# Workaround for lancedb 0.25.3 occasional SIGABRT during interpreter teardown after writes.
os._exit(0)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB query logging audit task: pre-seeded 200-row 64-d articles table; candidate implements LoggedSearcher in /home/user/myproject/solution.py."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3 for task + verifier
#   - /home/user/myproject/data.lancedb: pre-seeded LanceDB database with `articles` table (200 rows, 64-d embeddings)
