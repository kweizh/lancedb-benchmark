FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install a local, in-container MinIO server and the mc client (arch-aware).
RUN ARCH=$(dpkg --print-architecture) && \
    curl -fsSL "https://dl.min.io/server/minio/release/linux-${ARCH}/minio" -o /usr/local/bin/minio && \
    curl -fsSL "https://dl.min.io/client/mc/release/linux-${ARCH}/mc" -o /usr/local/bin/mc && \
    chmod +x /usr/local/bin/minio /usr/local/bin/mc && \
    mkdir -p /data/minio

# Bake deterministic input fixtures into /app/fixtures.
RUN mkdir -p /app/fixtures && cat >/tmp/gen_fixtures.py <<'PYEOF'
import json
import os

import numpy as np

os.makedirs("/app/fixtures", exist_ok=True)
DIM = 8
rng = np.random.default_rng(2026)

base_cats = ["news", "blog", "legacy", "wiki"]
added_cats = ["news", "blog", "wiki"]


def rand_vec():
    # components in [0.2, 0.8] keep every random row well away from the axis anchors
    return [round(float(x), 6) for x in (0.2 + 0.6 * rng.random(DIM))]


base = []
for i in range(40):
    base.append({
        "id": i,
        "text": f"base document {i}",
        "category": base_cats[i % 4],
        "vector": rand_vec(),
    })

added = []
for j, i in enumerate(range(40, 50)):
    added.append({
        "id": i,
        "text": f"added document {i}",
        "category": added_cats[j % 3],
        "vector": rand_vec(),
    })

# Query anchors: axis-aligned unit vectors.
q_added = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
q_deleted = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Anchor row for q_deleted: id 6 is a 'legacy' base row (6 % 4 == 2) -> deleted later.
assert base[6]["id"] == 6 and base[6]["category"] == "legacy"
base[6]["vector"] = list(q_deleted)

# Anchor row for q_added: id 45 is an appended row (survives the delete).
assert added[5]["id"] == 45 and added[5]["category"] != "legacy"
added[5]["vector"] = list(q_added)

with open("/app/fixtures/base.json", "w") as f:
    json.dump(base, f, indent=2)
with open("/app/fixtures/added.json", "w") as f:
    json.dump(added, f, indent=2)
with open("/app/fixtures/queries.json", "w") as f:
    json.dump({"q_added": q_added, "q_deleted": q_deleted}, f, indent=2)


# --- Build-time sanity: time travel must be observable ------------------------
def topk(rows, q, k=5):
    q = np.asarray(q, dtype=np.float64)
    scored = sorted(
        ((float(np.sum((np.asarray(r["vector"], dtype=np.float64) - q) ** 2)), r["id"])
         for r in rows),
        key=lambda t: (t[0], t[1]),
    )
    return [rid for _, rid in scored[:k]]


combined = base + added
deleted = [r for r in combined if r["category"] != "legacy"]

assert topk(base, q_added) != topk(combined, q_added), "q_added must differ base vs added"
assert topk(combined, q_added)[0] == 45, "q_added nearest in 'added' must be id 45"
assert 45 not in topk(base, q_added), "id 45 must be absent from 'base' snapshot"

assert topk(combined, q_deleted) != topk(deleted, q_deleted), "q_deleted must differ added vs deleted"
assert topk(combined, q_deleted)[0] == 6, "q_deleted nearest in 'added' must be legacy id 6"
assert 6 not in topk(deleted, q_deleted), "id 6 must be absent from 'deleted' snapshot"

n_legacy = sum(1 for r in combined if r["category"] == "legacy")
assert n_legacy == 10, f"expected 10 legacy rows, got {n_legacy}"
print("fixtures generated OK; legacy rows:", n_legacy)
PYEOF

RUN python3 /tmp/gen_fixtures.py && rm -f /tmp/gen_fixtures.py

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB OSS on a local in-container MinIO S3 backend; versioning/time-travel lakehouse task with baked deterministic fixtures."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: lancedb==0.25.3 (+ pyarrow, numpy, pandas) for the S3/versioning task
#   - MinIO server + mc client installed to /usr/local/bin (local, in-container S3 at 127.0.0.1:9000)
#   - entrypoint starts MinIO and creates the 'lancedb-lakehouse' bucket before the CMD
#   - /app/fixtures: deterministic base.json, added.json, queries.json inputs
#   - /home/user/myproject: empty project directory where the executor writes run.py

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
