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
RUN mkdir -p /home/user/project && mkdir -p /opt/zealt

# Inline seed script to build the LanceDB tables, dump the query vector,
# and capture ground-truth top-5 rankings for BLEND_ALPHA in {0.0, 1.0}.
RUN cat >/opt/zealt/seed.py <<'PYEOF'
import json
import os
import numpy as np
import pyarrow as pa
import lancedb

DATA_DIR = "/home/user/project/data"
PROJECT_DIR = "/home/user/project"
EXPECTED_DIR = "/opt/zealt"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(EXPECTED_DIR, exist_ok=True)

rng = np.random.default_rng(2026)
N_ITEMS = 80
DIM = 64
SEEN = list(range(10))  # user u_test's 10 most-recent items
K = 5

# 1. Generate item embeddings.
item_vecs = rng.standard_normal((N_ITEMS, DIM)).astype(np.float32)

# 2. Generate query vector from an independent stream so it is unrelated to taste.
query_vec = np.random.default_rng(20260).standard_normal(DIM).astype(np.float32)

# 3. Compute taste vector explicitly (verifier never reads this, but we need it
#    to capture the ground-truth ranking via LanceDB).
taste_vec = item_vecs[SEEN].mean(axis=0).astype(np.float32)

# 4. Persist the query vector for the candidate / verifier.
np.save(os.path.join(PROJECT_DIR, "query.npy"), query_vec)

# 5. Connect to LanceDB and create the items table.
db = lancedb.connect(DATA_DIR)

items_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("title", pa.string()),
    pa.field("category", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])
categories = ["electronics", "books", "fashion", "home", "sports"]
items_rows = [
    {
        "id": i,
        "title": f"Item-{i}",
        "category": categories[i % len(categories)],
        "vector": item_vecs[i].tolist(),
    }
    for i in range(N_ITEMS)
]
items_tbl = db.create_table("items", schema=items_schema, mode="overwrite")
items_tbl.add(items_rows)

# 6. Create the user_history table.
hist_schema = pa.schema([
    pa.field("user_id", pa.string()),
    pa.field("item_id", pa.int64()),
    pa.field("rating", pa.int32()),
    pa.field("ts", pa.int64()),  # microsecond epoch
])
base_ts = 1_700_000_000_000_000
hist_rows = []
# u_test: 10 history rows, strictly increasing timestamps, item 9 newest.
for idx, item_id in enumerate(SEEN):
    hist_rows.append({
        "user_id": "u_test",
        "item_id": int(item_id),
        "rating": 5,
        "ts": base_ts + idx * 1_000_000,
    })
# Some other users for realism.
for u_idx in range(3):
    uid = f"u_other_{u_idx}"
    for j in range(5):
        hist_rows.append({
            "user_id": uid,
            "item_id": int((u_idx * 7 + j + 11) % N_ITEMS),
            "rating": 3 + (j % 3),
            "ts": base_ts + (10_000 + j) * 1_000_000,
        })

hist_tbl = db.create_table("user_history", schema=hist_schema, mode="overwrite")
hist_tbl.add(hist_rows)

# 7. Capture ground-truth top-K rankings using LanceDB itself with a prefilter
#    NOT IN predicate. Using LanceDB-as-oracle removes any float32 / numpy
#    ordering ambiguity.
not_in_clause = "id NOT IN (" + ",".join(str(x) for x in SEEN) + ")"

res0 = (
    items_tbl.search(query_vec)
    .where(not_in_clause, prefilter=True)
    .limit(K)
    .to_pandas()
)
expected_alpha0 = [int(x) for x in res0["id"].tolist()]

res1 = (
    items_tbl.search(taste_vec)
    .where(not_in_clause, prefilter=True)
    .limit(K)
    .to_pandas()
)
expected_alpha1 = [int(x) for x in res1["id"].tolist()]

if expected_alpha0 == expected_alpha1:
    raise RuntimeError(
        "Fixture is malformed: alpha=0 and alpha=1 produce the same top-K. "
        "Adjust seeds so the rankings differ."
    )
if len(expected_alpha0) != K or len(expected_alpha1) != K:
    raise RuntimeError("Top-K rankings should have exactly K elements.")
if set(expected_alpha0) & set(SEEN) or set(expected_alpha1) & set(SEEN):
    raise RuntimeError("Seen items leaked into expected rankings; check fixture.")

with open(os.path.join(EXPECTED_DIR, "expected.json"), "w") as f:
    json.dump(
        {
            "expected_alpha0": expected_alpha0,
            "expected_alpha1": expected_alpha1,
            "seen_item_ids": SEEN,
        },
        f,
        indent=2,
    )

# Workaround for occasional SIGABRT during interpreter shutdown after a
# write-heavy LanceDB session (see lancedb 0.25.3 quirks in memory notes).
os._exit(0)
PYEOF

RUN python3 /opt/zealt/seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB personalized recommender task: items + user_history tables, precomputed query vector and ground-truth top-K rankings for alpha=0.0 and alpha=1.0."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/project: candidate workspace
#   - /home/user/project/data: prebuilt LanceDB database with `items` (80 x 64-d) and `user_history` (u_test has 10 rows + a few other users)
#   - /home/user/project/query.npy: seeded 64-d query vector
#   - /opt/zealt/expected.json: ground-truth top-5 rankings for alpha=0.0 and alpha=1.0, plus the seen item IDs
#   - /opt/zealt/seed.py: build-time seed script (kept for traceability)
