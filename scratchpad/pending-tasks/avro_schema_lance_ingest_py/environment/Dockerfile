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
        fastavro==1.9.7 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/avro_project && chmod 0755 /home/user/avro_project
RUN mkdir -p /app/data

# Build-time fixture: generate /app/data/records.avro with 300 deterministic
# records. Schema mirrors the contract described in the task:
#   - id: long
#   - title: string
#   - vector: array<float> (32 entries per row)
#   - metadata: record { author: string, tags: array<string>, score: double }
# Author is round-robin over 5 authors (alice/bob/carol/dave/eve), giving
# exactly 60 rows per author. Vectors come from a fixed numpy seed (2026)
# so brute-force top-k checks against any deterministic query vector are
# reproducible.
RUN cat >/tmp/generate_records_avro.py <<'PY'
import os

import fastavro
import numpy as np

OUT_PATH = "/app/data/records.avro"
N_ROWS = 300
DIM = 32
AUTHORS = ["alice", "bob", "carol", "dave", "eve"]
TAG_VOCAB = ["news", "tech", "science", "sports", "travel", "food", "music"]

SCHEMA = {
    "type": "record",
    "name": "Document",
    "namespace": "zealt.lancedb",
    "fields": [
        {"name": "id", "type": "long"},
        {"name": "title", "type": "string"},
        {"name": "vector", "type": {"type": "array", "items": "float"}},
        {
            "name": "metadata",
            "type": {
                "type": "record",
                "name": "Metadata",
                "fields": [
                    {"name": "author", "type": "string"},
                    {"name": "tags", "type": {"type": "array", "items": "string"}},
                    {"name": "score", "type": "double"},
                ],
            },
        },
    ],
}


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((N_ROWS, DIM)).astype(np.float32)
    scores = rng.uniform(0.0, 1.0, size=N_ROWS)

    records = []
    for i in range(N_ROWS):
        author = AUTHORS[i % len(AUTHORS)]
        n_tags = int(rng.integers(2, 4))  # 2 or 3
        tag_indices = rng.choice(len(TAG_VOCAB), size=n_tags, replace=False)
        tags = [TAG_VOCAB[int(j)] for j in tag_indices]
        record = {
            "id": int(i),
            "title": f"Document {i:04d}",
            "vector": [float(x) for x in vectors[i].tolist()],
            "metadata": {
                "author": author,
                "tags": tags,
                "score": float(scores[i]),
            },
        }
        records.append(record)

    parsed_schema = fastavro.parse_schema(SCHEMA)
    with open(OUT_PATH, "wb") as f:
        fastavro.writer(f, parsed_schema, records)

    # Sanity assertions baked into the build so a bad fixture fails fast.
    with open(OUT_PATH, "rb") as f:
        reader = fastavro.reader(f)
        read_records = list(reader)
    assert len(read_records) == N_ROWS, len(read_records)
    assert all(len(r["vector"]) == DIM for r in read_records)
    alice_count = sum(
        1 for r in read_records if r["metadata"]["author"] == "alice"
    )
    assert alice_count == N_ROWS // len(AUTHORS), alice_count
    print(f"Wrote {len(read_records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
PY
RUN python3 /tmp/generate_records_avro.py && \
    rm -f /tmp/generate_records_avro.py && \
    chmod 0644 /app/data/records.avro

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB Avro ingestion task: a 300-row Apache Avro fixture is pre-baked at /app/data/records.avro. The executor builds /home/user/avro_project/solution.py with ingest_avro(avro_path, table_name) that translates the Avro schema to Arrow (fixed_size_list<float32,32> + nested or flat metadata) and bulk-loads into LanceDB."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - LanceDB stack: lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - Apache Avro stack: fastavro==1.9.7
#   - /home/user/avro_project: empty project directory where the executor writes solution.py
#   - /app/data/records.avro: deterministic 300-record Avro fixture (seed=2026) with 5 authors (60 rows each)
