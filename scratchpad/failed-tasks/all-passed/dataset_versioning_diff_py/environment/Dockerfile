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
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /data/lancedb && chmod 0755 /data /data/lancedb
RUN mkdir -p /opt && chmod 0755 /opt

# Build-time seed script: materialise the customers table with deterministic
# version history (v1 create, v2 update, v3 delete, v4 add) and precompute
# ground-truth diffs to /opt/expected_diffs.json.
RUN cat >/tmp/seed_customers.py <<'PYEOF'
import json
import os
import numpy as np
import pyarrow as pa
import lancedb

SEED = 2026
DB_PATH = "/data/lancedb"
TABLE = "customers"
CATEGORIES = ["A", "B", "C", "D"]

rng = np.random.default_rng(SEED)

# --- v1: create_table with 100 rows ---
ids_initial = list(range(1, 101))
v1_rows = {}
init_payload = []
for i in ids_initial:
    price = round(float(rng.uniform(10.0, 1000.0)), 2)
    category = str(rng.choice(CATEGORIES))
    row = {"id": i, "name": f"customer_{i}", "price": price, "category": category}
    v1_rows[i] = dict(row)
    init_payload.append(row)

schema = pa.schema([
    ("id", pa.int64()),
    ("name", pa.string()),
    ("price", pa.float64()),
    ("category", pa.string()),
])

db = lancedb.connect(DB_PATH)
# Drop any stale table state first to keep build idempotent.
try:
    db.drop_table(TABLE)
except Exception:
    pass

table = db.create_table(TABLE, data=init_payload, schema=schema, mode="overwrite")
print("after create_table, versions:", len(table.list_versions()))

# --- v2: merge_insert to update 30 prices in a single transaction ---
updated_ids = sorted(int(x) for x in rng.choice(ids_initial, size=30, replace=False))
v2_rows = {k: dict(v) for k, v in v1_rows.items()}
upsert_payload = []
for uid in updated_ids:
    delta = round(float(rng.uniform(50.0, 250.0)), 2)
    new_price = round(v1_rows[uid]["price"] + delta, 2)
    v2_rows[uid]["price"] = new_price
    upsert_payload.append({
        "id": uid,
        "name": v2_rows[uid]["name"],
        "price": new_price,
        "category": v2_rows[uid]["category"],
    })
(
    table.merge_insert("id")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute(upsert_payload)
)
print("after merge_insert (v2), versions:", len(table.list_versions()))

# --- v3: delete 15 ids in a single delete() call ---
deleted_ids = sorted(int(x) for x in rng.choice(ids_initial, size=15, replace=False))
v3_rows = {k: dict(v) for k, v in v2_rows.items() if k not in set(deleted_ids)}
table.delete("id IN (" + ",".join(str(i) for i in deleted_ids) + ")")
print("after delete (v3), versions:", len(table.list_versions()))

# --- v4: add 20 brand-new rows ---
new_ids = list(range(101, 121))
add_payload = []
for nid in new_ids:
    price = round(float(rng.uniform(10.0, 1000.0)), 2)
    category = str(rng.choice(CATEGORIES))
    add_payload.append({
        "id": nid,
        "name": f"customer_{nid}",
        "price": price,
        "category": category,
    })
table.add(add_payload)
v4_rows = {k: dict(v) for k, v in v3_rows.items()}
for r in add_payload:
    v4_rows[r["id"]] = dict(r)
print("after add (v4), versions:", len(table.list_versions()))

# --- assert we ended up with exactly the planned 4 versions ---
versions = table.list_versions()
version_nums = [v.get("version") for v in versions]
assert len(versions) == 4, (
    "expected exactly 4 versions (create/update/delete/add); got "
    + str(len(versions))
    + " => "
    + repr(version_nums)
)
assert sorted(version_nums) == [1, 2, 3, 4], (
    "expected versions [1,2,3,4]; got " + repr(version_nums)
)

# --- sanity: spot-check the table at each historical version ---
table.checkout(1)
assert table.count_rows() == 100, "v1 should have 100 rows"
table.checkout(2)
assert table.count_rows() == 100, "v2 should still have 100 rows"
table.checkout(3)
assert table.count_rows() == 85, "v3 should have 85 rows"
table.checkout(4)
assert table.count_rows() == 105, "v4 should have 105 rows"
table.checkout_latest()


def compute_diff(a_rows, b_rows):
    a_ids = set(a_rows.keys())
    b_ids = set(b_rows.keys())
    added = sorted(int(i) for i in (b_ids - a_ids))
    removed = sorted(int(i) for i in (a_ids - b_ids))
    modified = []
    for i in sorted(a_ids & b_ids):
        a_row = {k: a_rows[i][k] for k in a_rows[i] if k != "id"}
        b_row = {k: b_rows[i][k] for k in b_rows[i] if k != "id"}
        if a_row != b_row:
            modified.append({"id": int(i), "old": a_row, "new": b_row})
    return {"added": added, "removed": removed, "modified": modified}


expected = {
    "1_4": compute_diff(v1_rows, v4_rows),
    "2_3": compute_diff(v2_rows, v3_rows),
    "3_4": compute_diff(v3_rows, v4_rows),
}

# Defensive shape assertions on the expected ground truth.
assert sorted(expected["3_4"]["added"]) == list(range(101, 121)), expected["3_4"]["added"]
assert expected["3_4"]["removed"] == [], expected["3_4"]["removed"]
assert expected["3_4"]["modified"] == [], expected["3_4"]["modified"]
assert expected["2_3"]["added"] == [], expected["2_3"]["added"]
assert sorted(expected["2_3"]["removed"]) == deleted_ids, expected["2_3"]["removed"]
assert expected["2_3"]["modified"] == [], expected["2_3"]["modified"]
assert sorted(expected["1_4"]["added"]) == list(range(101, 121)), expected["1_4"]["added"]
assert sorted(expected["1_4"]["removed"]) == deleted_ids, expected["1_4"]["removed"]
# modified count for v1->v4 = updates that survived the delete step
expected_modified_ids = sorted(set(updated_ids) - set(deleted_ids))
actual_modified_ids = sorted(int(r["id"]) for r in expected["1_4"]["modified"])
assert actual_modified_ids == expected_modified_ids, (
    "1_4 modified ids mismatch: " + repr(actual_modified_ids) + " vs " + repr(expected_modified_ids)
)

os.makedirs("/opt", exist_ok=True)
with open("/opt/expected_diffs.json", "w") as f:
    json.dump(expected, f, sort_keys=True)

print("SEED OK.")
print(
    "v1=", len(v1_rows),
    "v2=", len(v2_rows),
    "v3=", len(v3_rows),
    "v4=", len(v4_rows),
)
print("modified 1_4:", len(expected["1_4"]["modified"]))
print("modified 2_3:", len(expected["2_3"]["modified"]))
print("modified 3_4:", len(expected["3_4"]["modified"]))

# lancedb 0.25.3 occasionally SIGABRTs during interpreter teardown after a
# write-heavy session; exit hard to keep the docker build deterministic.
import os as _os
_os._exit(0)
PYEOF

RUN python3 /tmp/seed_customers.py && \
    test -f /opt/expected_diffs.json && \
    rm /tmp/seed_customers.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB customers table pre-seeded with 4 versions (create/update/delete/add) for dataset versioning diff evaluation."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack pinned for stability: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: empty project directory where the candidate writes solution.py
#   - /data/lancedb: LanceDB database containing the 'customers' table pre-seeded with 4 versions
#     * v1: create_table with 100 rows (id 1..100, name, price, category)
#     * v2: merge_insert updates the price of 30 rows
#     * v3: delete removes 15 rows
#     * v4: add 20 new rows (id 101..120)
#   - /opt/expected_diffs.json: precomputed ground-truth diffs for diff_versions(1,4), (2,3), (3,4)
