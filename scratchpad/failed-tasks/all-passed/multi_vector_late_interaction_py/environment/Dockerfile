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
RUN mkdir -p /home/user/myproject/lancedb_data

# Build-time seed script: deterministic fixture + LanceDB table.
RUN cat >/opt/seed_colbert.py <<'PYEOF'
import os
import numpy as np
import pyarrow as pa
import lancedb

DIM = 32
N_DOCS = 60
N_TOKENS = 4
RIGGED_DOC1 = 7
RIGGED_DOC2 = 42
DECOY_DOC_ID = 13
PROJECT_DIR = "/home/user/myproject"
LANCEDB_DIR = os.path.join(PROJECT_DIR, "lancedb_data")
FIXTURE_PATH = os.path.join(PROJECT_DIR, ".fixture.npz")
TABLE_NAME = "colbert_tokens"

rng = np.random.default_rng(2026)


def random_orthonormal(rng, k, d):
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    # Q has orthonormal columns of shape (d, k); rows of Q.T are orthonormal length-d vectors.
    return Q[:, :k].T.astype(np.float32)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def late_interaction(query, doc):
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    dn = doc / (np.linalg.norm(doc, axis=1, keepdims=True) + 1e-12)
    sims = qn @ dn.T
    return float(sims.max(axis=1).sum())


def sum_pool_cos(query, doc):
    return cos_sim(query.sum(axis=0), doc.sum(axis=0))


# Step 1: build the two multi-vector queries as orthonormal triples.
Q1 = random_orthonormal(rng, 3, DIM)  # (3, 32)
Q2 = random_orthonormal(rng, 3, DIM)  # (3, 32)

q1_sum = Q1.sum(axis=0)
q1_sum_dir = (q1_sum / np.linalg.norm(q1_sum)).astype(np.float32)

# Step 2: random distractor token vectors for every doc/token pair.
docs = rng.standard_normal((N_DOCS, N_TOKENS, DIM)).astype(np.float32) * 0.25

# Step 3: rig doc 7 for Q1 — three anchor tokens (one per query token) plus an
# adversarial fourth token that drags its sum-pool away from q1_sum_dir.
NOISE = 0.01
for i in range(3):
    docs[RIGGED_DOC1, i] = Q1[i] + rng.standard_normal(DIM).astype(np.float32) * NOISE
docs[RIGGED_DOC1, 3] = (-2.0 * q1_sum_dir).astype(np.float32) + rng.standard_normal(DIM).astype(np.float32) * NOISE

# Step 4: rig the decoy doc (13) for Q1 — all four tokens parallel to q1_sum_dir,
# so sum-pool maximally aligns but per-token MaxSim only reaches ~1/sqrt(3).
for j in range(N_TOKENS):
    docs[DECOY_DOC_ID, j] = q1_sum_dir + rng.standard_normal(DIM).astype(np.float32) * NOISE

# Step 5: rig doc 42 for Q2 — three anchor tokens plus a small-noise fourth.
for i in range(3):
    docs[RIGGED_DOC2, i] = Q2[i] + rng.standard_normal(DIM).astype(np.float32) * NOISE
docs[RIGGED_DOC2, 3] = rng.standard_normal(DIM).astype(np.float32) * 0.1

# Step 6: validate the rigging actually behaves as designed.
li_scores_q1 = np.array([late_interaction(Q1, docs[i]) for i in range(N_DOCS)])
sp_scores_q1 = np.array([sum_pool_cos(Q1, docs[i]) for i in range(N_DOCS)])
li_scores_q2 = np.array([late_interaction(Q2, docs[i]) for i in range(N_DOCS)])

top_li_q1 = int(np.lexsort((np.arange(N_DOCS), -li_scores_q1))[0])
top_sp_q1 = int(np.lexsort((np.arange(N_DOCS), -sp_scores_q1))[0])
top_li_q2 = int(np.lexsort((np.arange(N_DOCS), -li_scores_q2))[0])

print(f"[seed] late-interaction top-1 for Q1 = {top_li_q1} (expected {RIGGED_DOC1}); "
      f"score = {li_scores_q1[top_li_q1]:.4f}, runner-up = {sorted(li_scores_q1)[-2]:.4f}")
print(f"[seed] sum-pool top-1 for Q1 = {top_sp_q1} (expected {DECOY_DOC_ID}); "
      f"score = {sp_scores_q1[top_sp_q1]:.4f}, doc 7 sum-pool = {sp_scores_q1[RIGGED_DOC1]:.4f}")
print(f"[seed] late-interaction top-1 for Q2 = {top_li_q2} (expected {RIGGED_DOC2}); "
      f"score = {li_scores_q2[top_li_q2]:.4f}, runner-up = {sorted(li_scores_q2)[-2]:.4f}")

assert top_li_q1 == RIGGED_DOC1, (
    f"FIXTURE BUG: late-interaction top-1 for Q1 should be {RIGGED_DOC1}, got {top_li_q1}"
)
assert top_li_q2 == RIGGED_DOC2, (
    f"FIXTURE BUG: late-interaction top-1 for Q2 should be {RIGGED_DOC2}, got {top_li_q2}"
)
assert top_sp_q1 == DECOY_DOC_ID, (
    f"FIXTURE BUG: sum-pool top-1 for Q1 should be {DECOY_DOC_ID}, got {top_sp_q1}"
)
assert top_sp_q1 != RIGGED_DOC1, "FIXTURE BUG: sum-pool baseline must not be doc 7."

# Step 7: persist the fixture metadata for the verifier and (optionally) the candidate.
np.savez(
    FIXTURE_PATH,
    Q1=Q1.astype(np.float32),
    Q2=Q2.astype(np.float32),
    rigged_doc1=np.int64(RIGGED_DOC1),
    rigged_doc2=np.int64(RIGGED_DOC2),
    decoy_doc_id=np.int64(DECOY_DOC_ID),
)
print(f"[seed] wrote fixture to {FIXTURE_PATH}")

# Step 8: write the LanceDB table with explicit Arrow schema.
schema = pa.schema([
    pa.field("doc_id", pa.int64()),
    pa.field("token_idx", pa.int32()),
    pa.field("embedding", pa.list_(pa.float32(), DIM)),
])

doc_ids = []
token_idxs = []
embeddings = []
for d in range(N_DOCS):
    for t in range(N_TOKENS):
        doc_ids.append(d)
        token_idxs.append(t)
        embeddings.append(docs[d, t].tolist())

table_data = pa.Table.from_pydict(
    {
        "doc_id": pa.array(doc_ids, type=pa.int64()),
        "token_idx": pa.array(token_idxs, type=pa.int32()),
        "embedding": pa.array(embeddings, type=pa.list_(pa.float32(), DIM)),
    },
    schema=schema,
)

db = lancedb.connect(LANCEDB_DIR)
if TABLE_NAME in db.table_names():
    db.drop_table(TABLE_NAME)
tbl = db.create_table(TABLE_NAME, data=table_data)
print(f"[seed] created LanceDB table '{TABLE_NAME}' with {tbl.count_rows()} rows at {LANCEDB_DIR}")

# Avoid the lancedb 0.25.3 shutdown SIGABRT noted in prior tasks.
import os as _os
_os._exit(0)
PYEOF

RUN python3 /opt/seed_colbert.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB ColBERT-style multi-vector late-interaction fixture: 60 docs x 4 token vectors (dim 32) with rigged anchors for two multi-vector queries."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject/lancedb_data/colbert_tokens: 240-row LanceDB table seeded with numpy default_rng(2026)
#   - /home/user/myproject/.fixture.npz: Q1, Q2, rigged_doc1, rigged_doc2, decoy_doc_id for the verifier
