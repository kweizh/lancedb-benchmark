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
        fastapi==0.115.5 \
        "uvicorn[standard]==0.32.1" \
        "strawberry-graphql[fastapi]==0.262.4" \
        openai==1.54.5 \
        httpx==0.27.2 \
        pytest-xprocess \
        requests \
        tantivy==0.22.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject/data && chmod -R 0755 /home/user/myproject

RUN mkdir -p /opt && cat >/opt/_seed.py <<'PYEOF'
"""
Seed the LanceDB `documents_${ZEALT_RUN_ID}` table.

Idempotent: relies on the entrypoint marker file. This script unconditionally
recreates the table and writes /home/user/myproject/.seed_meta.json. Uses real
OpenAI text-embedding-3-small.
"""
import json
import os
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI


def main() -> int:
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("ZEALT_RUN_ID not set", file=sys.stderr)
        return 1
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    DATA_DIR = "/home/user/myproject/data"
    os.makedirs(DATA_DIR, exist_ok=True)
    table_name = f"documents_{run_id}"

    TOPICS = [
        "technology",
        "science",
        "art",
        "history",
        "sports",
        "music",
        "food",
        "travel",
        "nature",
        "space",
    ]
    TAGS = ["tech", "science", "art", "history", "sports"]
    BASE_TS = 1700000000  # 2023-11-14 ish

    N = 300
    bodies = []
    titles = []
    tags_per_row = []
    ts_per_row = []
    for i in range(N):
        topic = TOPICS[i % len(TOPICS)]
        marker = f"quokkamarker{i:04d}"
        title = f"Document {i}: {topic.capitalize()} overview"
        body = (
            f"Document number {i} discusses {topic}. "
            f"This passage explores {topic} concepts, vocabulary, and reasoning. "
            f"Notable identifier: {marker}. "
            f"Additional context helps the embedding model differentiate this row from others."
        )
        bodies.append(body)
        titles.append(title)
        tags_per_row.append([TAGS[i % len(TAGS)]])
        ts_per_row.append(BASE_TS + i * 86400)

    # Embed bodies with OpenAI in batches.
    client = OpenAI(api_key=api_key)
    vectors = np.zeros((N, 1536), dtype=np.float32)
    BATCH = 100
    for start in range(0, N, BATCH):
        chunk = bodies[start : start + BATCH]
        attempts = 0
        while True:
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small", input=chunk
                )
                break
            except Exception as exc:
                attempts += 1
                if attempts >= 5:
                    print(f"OpenAI embedding failed after retries: {exc}", file=sys.stderr)
                    return 2
                time.sleep(2 * attempts)
        for j, item in enumerate(resp.data):
            vectors[start + j] = np.asarray(item.embedding, dtype=np.float32)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("body", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("published_at", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), 1536)),
        ]
    )
    rows = pa.Table.from_pydict(
        {
            "id": list(range(N)),
            "title": titles,
            "body": bodies,
            "tags": tags_per_row,
            "published_at": ts_per_row,
            "vector": vectors.tolist(),
        },
        schema=schema,
    )

    db = lancedb.connect(DATA_DIR)
    if table_name in db.table_names():
        db.drop_table(table_name)
    tbl = db.create_table(table_name, rows)
    # Native Lance FTS index on body (no tantivy).
    tbl.create_fts_index("body", use_tantivy=False, replace=True)

    meta = {
        "table_name": table_name,
        "row_count": N,
        "row0_body": bodies[0],
        "row42_sentinel": f"quokkamarker0042",
        "base_ts": BASE_TS,
        "tags": TAGS,
    }
    with open("/home/user/myproject/.seed_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Seeded table `{table_name}` with {N} rows.")
    # Avoid lancedb 0.25.3 teardown SIGABRT during interpreter shutdown.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
PYEOF

RUN cat >/opt/entrypoint.sh <<'SHEOF'
#!/bin/bash
set -e
if [ -z "${ZEALT_RUN_ID}" ]; then
  echo "ZEALT_RUN_ID is not set; refusing to seed." >&2
  exit 1
fi
MARKER="/home/user/myproject/.seeded_${ZEALT_RUN_ID}"
if [ ! -f "${MARKER}" ]; then
  cd /home/user/myproject
  python3 /opt/_seed.py
  touch "${MARKER}"
fi
SHEOF
RUN chmod +x /opt/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + FastAPI/Strawberry GraphQL endpoint task; entrypoint seeds 300-row OpenAI-embedded documents table."

CMD ["/bin/bash", "-c", "trap : TERM INT; /opt/entrypoint.sh && sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3 packages: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#       fastapi==0.115.5 uvicorn[standard]==0.32.1 strawberry-graphql[fastapi]==0.262.4
#       openai==1.54.5 httpx==0.27.2 pytest-xprocess requests tantivy==0.22.0
#   - /opt/_seed.py: seeds documents_${ZEALT_RUN_ID} with 300 rows + 1536-d OpenAI embeddings
#   - /opt/entrypoint.sh: idempotent seeder gated by /home/user/myproject/.seeded_${ZEALT_RUN_ID}
#   - /home/user/myproject: project root; candidate writes app.py here
#   - /home/user/myproject/data: LanceDB database directory
