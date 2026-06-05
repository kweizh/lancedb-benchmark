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
        openai==1.54.5 \
        httpx==0.27.2 \
        tiktoken && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /home/user/myproject/data /app

RUN cat >/app/_seed.py <<'PYEOF'
"""Seed the LanceDB `products_${ZEALT_RUN_ID}` table.

Runs at container entrypoint because OPENAI_API_KEY and ZEALT_RUN_ID are only
available at runtime. Idempotent via a marker file.
"""

import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import lancedb
from openai import OpenAI


PROJECT_DIR = "/home/user/myproject"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "lancedb")
EXPECTED_PATH = os.path.join(PROJECT_DIR, ".expected.json")

RUN_ID = os.environ.get("ZEALT_RUN_ID")
if not RUN_ID:
    print("ZEALT_RUN_ID is not set", file=sys.stderr)
    sys.exit(1)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set", file=sys.stderr)
    sys.exit(1)

TABLE_NAME = f"products_{RUN_ID}"


# ---------------------------------------------------------------------------
# Anchor rows (ids 1..8) — every anchor query targets one of these uniquely.
# ---------------------------------------------------------------------------

ANCHORS = [
    {
        "id": 1,
        "sku": "LOW-001",
        "name": "Budget Pocket Accessory",
        "description": "Standard household basic affordable accessory product entry.",
        "category": "electronics",
        "price": 9.99,
        "release_date": "2023-05-10",
        "_query": "products under $50",
    },
    {
        "id": 2,
        "sku": "ABC-123",
        "name": "Specialty Widget ABC-123",
        "description": "Widget unit ABC-123 with proprietary connector port for specialty industrial applications.",
        "category": "electronics",
        "price": 199.99,
        "release_date": "2024-03-15",
        "_query": "find item ABC-123",
    },
    {
        "id": 3,
        "sku": "RUN-003",
        "name": "Trailblazer Marathon Pro",
        "description": "Comfortable cushioned running shoes engineered specifically for marathon training and daily long-distance jogging routines.",
        "category": "sports",
        "price": 129.99,
        "release_date": "2024-06-01",
        "_query": "comfortable running shoes",
    },
    {
        "id": 4,
        "sku": "RED-004",
        "name": "Crimson Streetwalker",
        "description": "Vibrant red colored athletic shoes featuring bold streetwear aesthetics for fashion city wear.",
        "category": "clothing",
        "price": 79.99,
        "release_date": "2024-08-20",
        "_query": "red shoes under $100",
    },
    {
        "id": 5,
        "sku": "CLO-005",
        "name": "Designer Linen Jacket",
        "description": "Designer linen jacket in elegant minimalist style for formal occasions and business events.",
        "category": "clothing",
        "price": 299.99,
        "release_date": "2024-02-10",
        "_query": "items in clothing category between $200 and $500",
    },
    {
        "id": 6,
        "sku": "XYZ-789",
        "name": "Reference Catalog XYZ-789",
        "description": "Rare collectors edition catalog code XYZ-789 reference volume for archivists.",
        "category": "books",
        "price": 89.99,
        "release_date": "2024-11-01",
        "_query": "search for code XYZ-789",
    },
    {
        "id": 7,
        "sku": "EAR-007",
        "name": "Skyfly Buds",
        "description": "Wireless bluetooth earbuds with active noise cancellation and long battery life for travelers.",
        "category": "electronics",
        "price": 149.99,
        "release_date": "2024-09-15",
        "_query": "wireless bluetooth earbuds",
    },
    {
        "id": 8,
        "sku": "WAL-008",
        "name": "Heritage Bifold Wallet",
        "description": "Premium genuine leather wallet handcrafted with elegant stitching and brass hardware.",
        "category": "clothing",
        "price": 199.99,
        "release_date": "2024-07-22",
        "_query": "premium leather wallet under $300",
    },
]

# Filler categories deliberately exclude `clothing` so anchor 5 stays unique.
FILLER_CATEGORIES = ["electronics", "home", "sports", "books", "food"]


def build_rows():
    rows = []
    for a in ANCHORS:
        rows.append({k: v for k, v in a.items() if not k.startswith("_")})
    rng = np.random.default_rng(2026)
    for i in range(9, 501):
        cat = FILLER_CATEGORIES[(i - 9) % len(FILLER_CATEGORIES)]
        # All prices >= 50 so anchor #1 (price<50) only matches LOW-001.
        price = float(50.0 + ((i * 7 + 13) % 500)) + 0.99
        # release_date varies a bit
        month = ((i % 12) + 1)
        release_date = f"2023-{month:02d}-15"
        rows.append({
            "id": int(i),
            "sku": f"PRD-{i:04d}",
            "name": f"Standard {cat.title()} Item {i:04d}",
            "description": f"Standard {cat} product number {i:04d} for general retail use and inventory stock.",
            "category": cat,
            "price": price,
            "release_date": release_date,
        })
    return rows


def embed_descriptions(client, rows, batch_size=100):
    vectors = [None] * len(rows)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        inputs = [r["description"] for r in chunk]
        for attempt in range(6):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=inputs,
                )
                break
            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                print(f"embedding batch {start} failed: {exc}; retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        else:
            raise RuntimeError("OpenAI embedding API failed after retries")
        for j, datum in enumerate(resp.data):
            vec = np.array(datum.embedding, dtype=np.float32)
            assert vec.shape == (1536,), f"expected 1536-d, got {vec.shape}"
            vectors[start + j] = vec
    return vectors


def main():
    marker = os.path.join(PROJECT_DIR, f".seeded_{RUN_ID}")
    if os.path.exists(marker):
        print(f"[seed] marker {marker} exists; skipping seeding.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    rows = build_rows()
    assert len(rows) == 500, f"expected 500 rows, got {len(rows)}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"[seed] embedding {len(rows)} descriptions via text-embedding-3-small ...")
    vectors = embed_descriptions(client, rows)

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("sku", pa.string()),
        pa.field("name", pa.string()),
        pa.field("description", pa.string()),
        pa.field("category", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("release_date", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1536)),
    ])

    arrow_rows = {
        "id": pa.array([r["id"] for r in rows], type=pa.int64()),
        "sku": pa.array([r["sku"] for r in rows], type=pa.string()),
        "name": pa.array([r["name"] for r in rows], type=pa.string()),
        "description": pa.array([r["description"] for r in rows], type=pa.string()),
        "category": pa.array([r["category"] for r in rows], type=pa.string()),
        "price": pa.array([r["price"] for r in rows], type=pa.float64()),
        "release_date": pa.array([r["release_date"] for r in rows], type=pa.string()),
        "vector": pa.array([v.tolist() for v in vectors], type=pa.list_(pa.float32(), 1536)),
    }
    tbl_arrow = pa.Table.from_pydict(arrow_rows, schema=schema)

    db = lancedb.connect(DATA_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=tbl_arrow)

    print("[seed] creating native FTS index on description ...")
    tbl.create_fts_index("description", use_tantivy=False)

    # Persist anchor expectations for the verifier
    anchors_payload = {
        "anchors": [
            {"query": a["_query"], "target_id": int(a["id"]), "expected_mode": expected_mode}
            for a, expected_mode in zip(
                ANCHORS,
                [
                    "filter_only",  # 1
                    "fts",          # 2
                    "vector",       # 3
                    "hybrid",       # 4
                    "filter_only",  # 5
                    "fts",          # 6
                    "vector",       # 7
                    "hybrid",       # 8
                ],
            )
        ]
    }
    with open(EXPECTED_PATH, "w") as f:
        json.dump(anchors_payload, f, indent=2)

    with open(marker, "w") as f:
        f.write("ok\n")

    print(f"[seed] done; wrote {TABLE_NAME}, anchors {EXPECTED_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Avoid lancedb 0.25.3 SIGABRT during interpreter teardown.
        os._exit(0)
PYEOF

RUN cat >/usr/local/bin/entrypoint.sh <<'SHEOF'
#!/usr/bin/env bash
set -e
RUN_ID="${ZEALT_RUN_ID:-default}"
MARKER="/home/user/myproject/.seeded_${RUN_ID}"
if [ ! -f "$MARKER" ]; then
    python3 /app/_seed.py
fi
exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
SHEOF

RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB query router with OpenAI structured outputs: seeds 500-row products table with text-embedding-3-small 1536-d vectors and native FTS index on description at entrypoint."

CMD ["/usr/local/bin/entrypoint.sh"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, openai==1.54.5, tiktoken
#   - /app/_seed.py: idempotent entrypoint seed script (500 products + 1536-d OpenAI vectors + native FTS index on description)
#   - /usr/local/bin/entrypoint.sh: runs seed once per ZEALT_RUN_ID, then sleeps
#   - /home/user/myproject: candidate working dir; /home/user/myproject/data/lancedb holds the LanceDB store
