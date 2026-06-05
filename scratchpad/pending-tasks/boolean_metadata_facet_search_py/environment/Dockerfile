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
RUN mkdir -p /home/user/myproject /home/user/myproject/data

# ---- Seed script: builds the products_${ZEALT_RUN_ID} table at container start ----
RUN cat >/usr/local/bin/_seed.py <<'PYEOF'
import os
import sys
import numpy as np
import pyarrow as pa
import lancedb

DB_DIR = "/home/user/myproject/data/lancedb"
RUN_ID = os.environ.get("ZEALT_RUN_ID")
if not RUN_ID:
    print("ZEALT_RUN_ID is not set; aborting seed.", file=sys.stderr)
    sys.exit(1)

TABLE = f"products_{RUN_ID}"

BRANDS = [f"Brand{c}" for c in "ABCDEFGHIJ"]
CATEGORIES = [f"Cat{i+1}" for i in range(6)]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]

os.makedirs(DB_DIR, exist_ok=True)
db = lancedb.connect(DB_DIR)

existing = set(db.table_names())
if TABLE in existing:
    print(f"Table {TABLE} already present; skipping seed.")
    os._exit(0)

N = 1000
DIM = 32
rng = np.random.default_rng(2026)

# Draw vectors FIRST to anchor RNG (per repository convention)
vectors = rng.standard_normal((N, DIM)).astype(np.float32)
brand_idx = rng.integers(0, len(BRANDS), size=N)
category_idx = rng.integers(0, len(CATEGORIES), size=N)
color_idx = rng.integers(0, len(COLORS), size=N)
in_stock = rng.random(N) < 0.6
prices = rng.uniform(5.0, 500.0, size=N).astype(np.float64)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("brand", pa.string()),
    pa.field("category", pa.string()),
    pa.field("color", pa.string()),
    pa.field("in_stock", pa.bool_()),
    pa.field("price", pa.float64()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

ids = list(range(N))
data = {
    "id": pa.array(ids, type=pa.int64()),
    "brand": pa.array([BRANDS[i] for i in brand_idx], type=pa.string()),
    "category": pa.array([CATEGORIES[i] for i in category_idx], type=pa.string()),
    "color": pa.array([COLORS[i] for i in color_idx], type=pa.string()),
    "in_stock": pa.array(in_stock.tolist(), type=pa.bool_()),
    "price": pa.array(prices.tolist(), type=pa.float64()),
    "vector": pa.array([v.tolist() for v in vectors], type=pa.list_(pa.float32(), DIM)),
}
tbl_arrow = pa.table(data, schema=schema)

db.create_table(TABLE, tbl_arrow, mode="overwrite")

# Sanity invariants — fail fast if seed is broken
df = db.open_table(TABLE).to_pandas()
assert len(df) == N, f"Seed row count mismatch: {len(df)}"
assert df["brand"].nunique() == len(BRANDS), "Not all 10 brands seeded"
assert df["category"].nunique() == len(CATEGORIES), "Not all 6 categories seeded"
assert df["color"].nunique() == len(COLORS), "Not all 8 colors seeded"
assert len(df["vector"].iloc[0]) == DIM, "Vector dim mismatch"

print(f"Seeded {TABLE} with {N} rows.")
os._exit(0)
PYEOF

# ---- Entrypoint: idempotent seed via marker, then sleep ----
RUN cat >/usr/local/bin/entrypoint.sh <<'BASHEOF'
#!/bin/bash
set -u
RUN_ID="${ZEALT_RUN_ID:-}"
if [ -z "${RUN_ID}" ]; then
    echo "ZEALT_RUN_ID is not set" >&2
    exit 1
fi
MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "${MARKER}" ]; then
    python3 /usr/local/bin/_seed.py
    touch "${MARKER}"
fi
trap : TERM INT
sleep infinity & wait
BASHEOF
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB faceted product search task — boolean metadata filters + group-by facet counts."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#   - /home/user/myproject/data/lancedb: LanceDB store seeded at first container start
#   - /home/user/myproject/data/lancedb/products_${ZEALT_RUN_ID}: 1000-row product table with
#       fields {id int64, brand str (10 vals), category str (6 vals), color str (8 vals),
#               in_stock bool, price float64, vector fixed_size_list<float32, 32>}
#   - /usr/local/bin/_seed.py: deterministic seed script (numpy seed=2026)
#   - /usr/local/bin/entrypoint.sh: idempotent run-id-aware seeding entrypoint
