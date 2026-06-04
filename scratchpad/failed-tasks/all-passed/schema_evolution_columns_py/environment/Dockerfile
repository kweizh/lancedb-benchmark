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

RUN mkdir -p /workspace/solution /workspace/output /workspace/db

# Seed the inventory table with the deterministic fixture (12 rows,
# columns: id int64, sku string, qty int32, vector fixed_size_list<float32>[4]).
RUN cat >/tmp/seed_inventory.py <<'PYEOF' && \
    python3 /tmp/seed_inventory.py && \
    rm -f /tmp/seed_inventory.py
import os

import lancedb
import numpy as np
import pyarrow as pa

DB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "inventory"

# Deterministic seed fixture.
qty_values = [3, 12, 1, 7, 8, 4, 15, 2, 10, 5, 0, 6]
ids = list(range(1, len(qty_values) + 1))
skus = [f"SKU-{i:03d}" for i in ids]

# Deterministic 4-d float32 vectors (not used by the task, will be dropped).
rng = np.random.default_rng(42)
vectors = rng.random((len(ids), 4), dtype=np.float32)

schema = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("sku", pa.string(), nullable=False),
    pa.field("qty", pa.int32(), nullable=False),
    pa.field(
        "vector",
        pa.list_(pa.field("item", pa.float32(), nullable=True), 4),
        nullable=False,
    ),
])

vector_array = pa.FixedSizeListArray.from_arrays(
    pa.array(vectors.reshape(-1).tolist(), type=pa.float32()),
    list_size=4,
)

table_data = pa.Table.from_arrays(
    [
        pa.array(ids, type=pa.int64()),
        pa.array(skus, type=pa.string()),
        pa.array(qty_values, type=pa.int32()),
        vector_array,
    ],
    schema=schema,
)

os.makedirs(DB_URI, exist_ok=True)
db = lancedb.connect(DB_URI)
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)
db.create_table(TABLE_NAME, table_data)

# Sanity check.
tbl = db.open_table(TABLE_NAME)
assert tbl.count_rows() == len(ids), tbl.count_rows()
field_names = {f.name for f in tbl.schema}
assert field_names == {"id", "sku", "qty", "vector"}, field_names
print("Seeded inventory table with", tbl.count_rows(), "rows.")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB schema-evolution task: pre-seeded 'inventory' table at /workspace/db; candidate adds derived columns and drops the vector column via lancedb.Table.add_columns / drop_columns."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB Python SDK (lancedb==0.25.3) plus pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /workspace/solution: directory where the candidate writes main.py
#   - /workspace/output:   directory for the candidate's schema_after.json artifact
#   - /workspace/db:       LanceDB database directory with the pre-seeded 'inventory' table
