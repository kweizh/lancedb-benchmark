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
RUN mkdir -p /app/source

# Bake the deterministic Arrow IPC stream and query vector at build time.
RUN python3 - <<'PYEOF'
import os
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

rng = np.random.default_rng(2026)
N = 2000
DIM = 48
TAG_POOL = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]

ids = pa.array(np.arange(N, dtype=np.int64), type=pa.int64())
texts = pa.array([f"event_{i:04d}_payload" for i in range(N)], type=pa.string())

tags_list = []
for _ in range(N):
    k = int(rng.integers(0, 4))  # 0..3 tags per row
    if k == 0:
        chosen = []
    else:
        chosen = rng.choice(TAG_POOL, size=k, replace=False).tolist()
    tags_list.append(chosen)
tags_arr = pa.array(tags_list, type=pa.list_(pa.string()))

emb = rng.standard_normal(size=(N, DIM)).astype(np.float32)
flat = pa.array(emb.reshape(-1).tolist(), type=pa.float32())
emb_arr = pa.FixedSizeListArray.from_arrays(flat, DIM)

base_ms = 1_700_000_000_000  # ~2023-11-14T22:13:20Z
offsets = rng.integers(0, 365 * 24 * 60 * 60 * 1000, size=N).astype(np.int64)
ts = pa.array((base_ms + offsets).astype(np.int64), type=pa.timestamp("ms"))

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("tags", pa.list_(pa.string())),
    pa.field("embedding", pa.list_(pa.float32(), DIM)),
    pa.field("created_at", pa.timestamp("ms")),
])

table = pa.Table.from_arrays([ids, texts, tags_arr, emb_arr, ts], schema=schema)

os.makedirs("/app/source", exist_ok=True)
with pa.OSFile("/app/source/dataset.arrows", "wb") as sink:
    with ipc.new_stream(sink, schema) as writer:
        chunk = 500
        offset = 0
        while offset < N:
            sub = table.slice(offset, min(chunk, N - offset))
            for b in sub.to_batches():
                writer.write_batch(b)
            offset += chunk

qv = rng.standard_normal(size=(DIM,)).astype(np.float32)
np.save("/app/query_vector.npy", qv)

# Sanity round-trip check at build time.
with pa.OSFile("/app/source/dataset.arrows", "rb") as src:
    with ipc.open_stream(src) as reader:
        round_trip = reader.read_all()
assert round_trip.num_rows == N, f"expected {N} rows, got {round_trip.num_rows}"
assert round_trip.schema.equals(schema, check_metadata=False), "schema round-trip mismatch"
print(f"[build] wrote {round_trip.num_rows} rows to /app/source/dataset.arrows; query vector dim={qv.shape[0]}")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB task environment for migrating an Arrow IPC stream into a LanceDB table with a top-5 vector search."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - /home/user/myproject: empty project directory the candidate must populate with run.py
#   - /app/source/dataset.arrows: deterministic 2000-row Arrow IPC stream (id, text, tags, fixed_size_list<float32, 48> embedding, timestamp[ms] created_at)
#   - /app/query_vector.npy: precomputed (48,) float32 query vector
