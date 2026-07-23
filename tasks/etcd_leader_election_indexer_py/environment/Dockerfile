FROM ubuntu:24.04

ARG POCHI_VERSION

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep procps && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -fsSL https://getpochi.com/install.sh | bash -s ${POCHI_VERSION} && \
    ln -s /root/.pochi/bin/pochi /usr/local/bin/pochi && \
    pip3 install --break-system-packages \
        pytest-json-ctrf requests \
        lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install a local etcd v3 server + etcdctl (no external services at runtime).
RUN ETCD_VER=v3.5.17 && \
    ARCH=$(dpkg --print-architecture) && \
    curl -fsSL -o /tmp/etcd.tar.gz \
        https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz && \
    tar xzf /tmp/etcd.tar.gz -C /tmp && \
    cp /tmp/etcd-${ETCD_VER}-linux-${ARCH}/etcd /usr/local/bin/etcd && \
    cp /tmp/etcd-${ETCD_VER}-linux-${ARCH}/etcdctl /usr/local/bin/etcdctl && \
    rm -rf /tmp/etcd.tar.gz /tmp/etcd-${ETCD_VER}-linux-${ARCH} && \
    mkdir -p /var/lib/etcd

# Idempotent helper that starts the local etcd server if it is not already up.
RUN cat >/usr/local/bin/start-etcd.sh <<'SH' && chmod +x /usr/local/bin/start-etcd.sh
#!/bin/bash
if curl -fsS http://127.0.0.1:2379/health >/dev/null 2>&1; then
    exit 0
fi
nohup etcd \
    --data-dir /var/lib/etcd \
    --listen-client-urls http://127.0.0.1:2379 \
    --advertise-client-urls http://127.0.0.1:2379 \
    >/var/log/etcd.log 2>&1 &
for _ in $(seq 1 40); do
    if curl -fsS http://127.0.0.1:2379/health >/dev/null 2>&1; then
        exit 0
    fi
    sleep 0.5
done
echo "etcd failed to become ready" >&2
exit 1
SH

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Seed-fixture module: (re)creates the shared LanceDB `documents` table with a
# trained IVF_PQ vector index over 300 rows plus 200 freshly appended, not-yet
# indexed rows. Also used by the verifier to reset / extend the dataset.
RUN cat >/home/user/myproject/seed_dataset.py <<'PYEOF'
import os
from datetime import timedelta

import lancedb
import numpy as np
import pyarrow as pa

DB_URI = "/home/user/myproject/lancedb"
TABLE_NAME = "documents"
EMBED_DIM = 16
SEED = 2026

SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ]
)


def _make_rows(start, count, rng):
    rows = []
    for i in range(start, start + count):
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        rows.append({"id": int(i), "text": "document-%d" % i, "vector": vec.tolist()})
    return rows


def _index_name(tbl):
    idxs = tbl.list_indices()
    return idxs[0].name if idxs else "vector_idx"


def reset():
    db = lancedb.connect(DB_URI)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    rng = np.random.default_rng(SEED)
    initial = _make_rows(0, 300, rng)
    tbl = db.create_table(TABLE_NAME, data=initial, schema=SCHEMA)
    tbl.create_index(
        index_type="IVF_PQ",
        metric="cosine",
        num_partitions=8,
        num_sub_vectors=4,
        replace=True,
    )
    tbl.wait_for_index([_index_name(tbl)], timedelta(seconds=180))
    # 200 unindexed rows appended in small batches -> multiple small fragments.
    for start in range(300, 500, 20):
        tbl.add(_make_rows(start, 20, rng))
    return tbl.count_rows()


def append_unindexed(n):
    db = lancedb.connect(DB_URI)
    tbl = db.open_table(TABLE_NAME)
    existing = tbl.count_rows()
    rng = np.random.default_rng(SEED + existing)
    added = 0
    while added < n:
        batch = min(20, n - added)
        tbl.add(_make_rows(existing + added, batch, rng))
        added += batch
    return n


if __name__ == "__main__":
    total = reset()
    print("seeded", total, "rows")
    os._exit(0)
PYEOF

RUN cd /home/user/myproject && python3 seed_dataset.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + local etcd v3 environment for a distributed indexer leader-election task."

CMD ["/bin/bash", "-c", "/usr/local/bin/start-etcd.sh; trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep procps
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3, plus requests for the etcd JSON gateway
#   - etcd v3.5.17 server + etcdctl in /usr/local/bin, data dir /var/lib/etcd
#   - /usr/local/bin/start-etcd.sh: idempotent local etcd starter (also launched by CMD)
#   - /home/user/myproject/seed_dataset.py: fixture builder (300 indexed + 200 unindexed rows)
#   - /home/user/myproject/lancedb: seeded `documents` table with a trained IVF_PQ index

# Setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
