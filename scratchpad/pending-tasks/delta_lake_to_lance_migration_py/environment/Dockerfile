FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        deltalake==0.20.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject
RUN mkdir -p /app/delta_data /app/lancedb_data

# Seed the Delta Lake table at build time:
#   v0: 500 rows with ids 0..499, deterministic categories from {A,B,C,D}
#   v1: append 300 rows with ids 500..799
#   v2: dt.update() flips category of ids 0..199 to 'Z'
RUN cat >/tmp/_seed_delta.py <<'PYEOF'
import numpy as np
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

DELTA_PATH = "/app/delta_data/products"
DIM = 32
CATS = ["A", "B", "C", "D"]
rng = np.random.default_rng(2026)


def build_table(ids, version_value):
    n = len(ids)
    names = [f"item_{int(i)}" for i in ids]
    cats = [CATS[int(i) % 4] for i in ids]
    vecs = rng.standard_normal((n, DIM)).astype(np.float32)
    # Variable-length list<float32> (NOT fixed_size_list) — the migration step
    # is responsible for converting to fixed_size_list<float32, 32>.
    vec_list = pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32()))
    return pa.table({
        "id": pa.array(ids.astype(np.int64), type=pa.int64()),
        "name": pa.array(names, type=pa.string()),
        "vector": vec_list,
        "category": pa.array(cats, type=pa.string()),
        "version": pa.array(np.full(n, version_value, dtype=np.int64), type=pa.int64()),
    })


# Version 0: initial 500 rows
ids0 = np.arange(0, 500, dtype=np.int64)
tbl0 = build_table(ids0, 0)
write_deltalake(DELTA_PATH, tbl0, mode="overwrite")

# Version 1: append +300 rows (ids 500..799)
ids1 = np.arange(500, 800, dtype=np.int64)
tbl1 = build_table(ids1, 1)
write_deltalake(DELTA_PATH, tbl1, mode="append")

# Version 2: update categories of the 200 rows where id < 200 to 'Z'
dt = DeltaTable(DELTA_PATH)
dt.update(updates={"category": "'Z'"}, predicate="id < 200")

# Sanity check
dt = DeltaTable(DELTA_PATH)
assert dt.version() == 2, f"Expected Delta version 2 after seeding, got {dt.version()}"
latest = dt.to_pyarrow_table().to_pandas()
assert len(latest) == 800, f"Expected 800 rows in latest snapshot, got {len(latest)}"
assert set(latest["id"].tolist()) == set(range(800))
changed = latest[latest["id"] < 200]
assert (changed["category"] == "Z").all(), "Expected ids 0..199 to have category 'Z' after v2."
unchanged = latest[(latest["id"] >= 200) & (latest["id"] < 500)]
assert all(c in CATS for c in unchanged["category"].tolist())

dt0 = DeltaTable(DELTA_PATH, version=0).to_pyarrow_table().to_pandas()
assert len(dt0) == 500
assert set(dt0["id"].tolist()) == set(range(500))
print("Delta seeding OK: versions=", [0, 1, 2])
PYEOF
RUN python3 /tmp/_seed_delta.py && rm /tmp/_seed_delta.py

LABEL Author="generate-harbor-tasks"
LABEL Description="Ubuntu 24.04 + lancedb 0.25.3 + deltalake 0.20.2 with a pre-seeded 3-version Delta Lake table at /app/delta_data/products for migration into LanceDB."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, deltalake==0.20.2
#   - /home/user/myproject: empty project directory for the candidate
#   - /app/delta_data/products: pre-seeded 3-version Delta Lake table (v0=500, v1=+300, v2=update id<200 → category 'Z')
#   - /app/lancedb_data: empty target LanceDB directory; candidate writes products_${ZEALT_RUN_ID} and migration_audit_${ZEALT_RUN_ID}
