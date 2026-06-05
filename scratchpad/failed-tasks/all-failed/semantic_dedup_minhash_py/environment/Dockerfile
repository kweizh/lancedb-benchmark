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
        "datasketch>=1.6,<2" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /app/lancedb_data

# Seed the LanceDB corpus and ground-truth pair list at build time.
# Build-time ZEALT_RUN_ID is "build"; the entrypoint re-seeds the table under
# the actual ZEALT_RUN_ID before the task starts so parallel runs stay isolated.
RUN cat >/app/seed.py <<'PYEOF'
import json
import os
import sys
import numpy as np
import pyarrow as pa
import lancedb

RUN_ID = os.environ.get("ZEALT_RUN_ID", "build")
TABLE_NAME = f"documents_{RUN_ID}"
LANCEDB_DIR = "/app/lancedb_data"
SEED = 2026
DIM = 64
NUM_PAIRS = 50
NUM_SINGLETONS = 200
TOTAL = NUM_PAIRS * 2 + NUM_SINGLETONS  # 300

rng = np.random.default_rng(SEED)

# Build a 2000-word base vocabulary so each document can sample ~80 unique words
# with very little cross-document overlap among unrelated docs.
VOCAB = [f"word{idx:05d}" for idx in range(2000)]

DOC_LENGTH = 200

def make_text(rng_local, doc_index, length=DOC_LENGTH):
    # Each document samples its words from a disjoint-ish slice of the vocab to
    # guarantee that random singleton pairs have very low Jaccard on 3-shingles.
    # Slice center moves with doc_index; slice width = 200 words.
    center = (doc_index * 37) % len(VOCAB)
    half = 100
    lo = max(0, center - half)
    hi = min(len(VOCAB), center + half)
    pool = VOCAB[lo:hi]
    words = rng_local.choice(pool, size=length, replace=True).tolist()
    return " ".join(words)

def mutate(text, rng_local, n_edits=4):
    # Apply a small fixed number of single-word substitutions, spaced out enough
    # that each substitution affects only ~3 of the 3-shingles. With n_edits=4 on
    # a 200-word document, ~12 shingles are perturbed out of 198 (~6%), which
    # leaves Jaccard >= 0.89 — comfortably above the LSH threshold of 0.7 and
    # well within MinHash(num_perm=128) recall.
    words = list(text.split())
    # Sample positions with minimum spacing so edits don't overlap on the same
    # shingle and reduce Jaccard further than the analytical estimate.
    candidates = list(range(2, len(words) - 2))
    rng_local.shuffle(candidates)
    chosen = []
    for p in candidates:
        if all(abs(p - q) >= 4 for q in chosen):
            chosen.append(p)
            if len(chosen) == n_edits:
                break
    for p in chosen:
        while True:
            cand = VOCAB[int(rng_local.integers(0, len(VOCAB)))]
            if cand != words[p]:
                words[p] = cand
                break
    return " ".join(words)

def unit(v):
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return (v / n).astype(np.float32)

rows = []
pairs = []

# IDs 0..49 are anchors of duplicate pairs.
# IDs 50..99 are clones (paired with id i-50).
# IDs 100..299 are 200 singletons.
for i in range(NUM_PAIRS):
    text_anchor = make_text(rng, doc_index=i, length=DOC_LENGTH)
    text_clone = mutate(text_anchor, rng, n_edits=4)

    base_vec = unit(rng.standard_normal(DIM).astype(np.float32))
    noise = rng.standard_normal(DIM).astype(np.float32) * 0.01
    clone_vec = unit(base_vec + noise)

    rows.append({"id": i, "text": text_anchor, "vector": base_vec.tolist()})
    rows.append({"id": NUM_PAIRS + i, "text": text_clone, "vector": clone_vec.tolist()})
    pairs.append([i, NUM_PAIRS + i])

# Singletons: ids 100..299, each with its own random text slice and random unit vector.
for j in range(NUM_SINGLETONS):
    doc_id = NUM_PAIRS * 2 + j  # 100..299
    text = make_text(rng, doc_index=100 + j, length=DOC_LENGTH)
    vec = unit(rng.standard_normal(DIM).astype(np.float32))
    rows.append({"id": doc_id, "text": text, "vector": vec.tolist()})

# Sanity-check distances at build time so a bad seed never silently produces a
# fixture that violates the verifier's assumptions.
ids = [r["id"] for r in rows]
vecs = np.array([r["vector"] for r in rows], dtype=np.float32)
# Cosine distance matrix between dup pairs.
for a, b in pairs:
    va = vecs[ids.index(a)]
    vb = vecs[ids.index(b)]
    cd = 1.0 - float(np.dot(va, vb))
    assert cd <= 0.05, f"Dup pair ({a},{b}) cosine distance {cd:.4f} > 0.05"

# Verify non-pair vectors are all > 0.10 cosine distance.
dup_set = {(min(a, b), max(a, b)) for a, b in pairs}
for i_idx in range(len(ids)):
    for j_idx in range(i_idx + 1, len(ids)):
        a, b = ids[i_idx], ids[j_idx]
        key = (min(a, b), max(a, b))
        if key in dup_set:
            continue
        cd = 1.0 - float(np.dot(vecs[i_idx], vecs[j_idx]))
        assert cd > 0.10, f"Non-pair ({a},{b}) cosine distance {cd:.4f} <= 0.10"

# Connect and create the LanceDB table.
db = lancedb.connect(LANCEDB_DIR)
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

# pyarrow Table-from-pylist conversion handles the list<float32> column cleanly.
arrow_tbl = pa.Table.from_pylist(rows, schema=schema)
tbl = db.create_table(TABLE_NAME, data=arrow_tbl)
assert tbl.count_rows() == TOTAL, f"Expected {TOTAL} rows, got {tbl.count_rows()}"

# Persist the ground-truth pair list so the verifier can read it without
# re-deriving the corpus.
with open("/app/ground_truth_pairs.json", "w", encoding="utf-8") as f:
    json.dump({"pairs": pairs}, f)

# Workaround the lancedb 0.25.3 interpreter-teardown SIGABRT seen with
# write-heavy seed scripts.
os._exit(0)
PYEOF

RUN ZEALT_RUN_ID=build python3 /app/seed.py

# Entrypoint re-seeds the LanceDB table under the actual run-id passed at task
# time, so parallel runs don't collide on table names and so the verifier can
# always rely on `documents_${ZEALT_RUN_ID}` existing.
RUN cat >/app/entrypoint.sh <<'SHEOF'
#!/bin/bash
set -u
if [ -n "${ZEALT_RUN_ID:-}" ] && [ "${ZEALT_RUN_ID}" != "build" ]; then
  python3 /app/seed.py >/app/seed.log 2>&1 || true
fi
exec "$@"
SHEOF
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + datasketch MinHash hybrid dedup task with a 300-row seeded corpus."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 datasketch>=1.6,<2
#   - /app/seed.py builds a deterministic 300-row LanceDB corpus (50 dup pairs + 200 singletons)
#     in /app/lancedb_data and writes /app/ground_truth_pairs.json
#   - /app/entrypoint.sh re-seeds the table under the real ZEALT_RUN_ID before the task starts
#   - /home/user/myproject is the empty project directory where the candidate writes solution.py + run.py
