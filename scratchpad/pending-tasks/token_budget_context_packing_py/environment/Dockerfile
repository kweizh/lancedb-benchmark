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
        tiktoken \
        httpx==0.27.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /app/lancedb_data

# Bake the seed script. It runs at container entrypoint, NOT at build time,
# because OPENAI_API_KEY is only available at runtime.
RUN cat >/app/_seed.py <<'PYEOF'
"""Seed the LanceDB 'chunks_${ZEALT_RUN_ID}' table with 150 rigged chunks.

Embeddings are computed at container entrypoint using the real OpenAI
text-embedding-3-small API (1536-d).  Rows have variable-length text and a
'topic_id' rigged so that the descending-relevance + topic-diversity (max 2
per topic) packer produces a different selection than the no-diversity packer
at max_tokens=600.  The script asserts both invariants before writing.
"""
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
import tiktoken
from openai import OpenAI

LANCEDB_DIR = "/app/lancedb_data"
RUN_ID = os.environ.get("ZEALT_RUN_ID")
if not RUN_ID:
    print("ZEALT_RUN_ID is not set; cannot seed.", file=sys.stderr)
    sys.exit(2)

if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set; cannot seed.", file=sys.stderr)
    sys.exit(2)

ENC = tiktoken.get_encoding("cl100k_base")
MODEL = "text-embedding-3-small"


def pad_to_tokens(text: str, target_min: int, filler: str) -> str:
    cur = text
    while len(ENC.encode(cur)) < target_min:
        cur = cur + " " + filler
    return cur


# 5 highly-relevant paraphrases of the canonical query.
RELEVANT_VDB = [
    ("Vector databases are designed to handle production retrieval pipelines by indexing "
     "high-dimensional embeddings and serving low-latency nearest-neighbor queries at scale. "
     "Modern systems combine ANN indexes such as IVF-PQ and HNSW with metadata filters, "
     "delivering subsecond responses for retrieval-augmented generation workloads."),
    ("In production retrieval pipelines, vector databases play a central role by storing "
     "embeddings produced by transformer encoders and providing approximate nearest neighbor "
     "search over millions of vectors. They expose tunable knobs for recall, throughput, and "
     "latency, and integrate with feature stores and orchestration frameworks."),
    ("Production-grade vector databases must support online ingestion, incremental indexing, "
     "and low-latency retrieval inside modern retrieval-augmented generation pipelines. "
     "Engineers balance memory pressure, compaction, and quantization while keeping recall "
     "above ninety-five percent on evaluation queries."),
    ("Modern vector databases optimize production retrieval pipelines through partitioning, "
     "product quantization, and inverted file indexes that route a query to a subset of "
     "probable centroids. This drastically reduces wall-clock latency while preserving recall "
     "for downstream retrieval-augmented generation services."),
    ("When deploying vector databases for production retrieval pipelines, platform teams focus "
     "on throughput, recall, and latency trade-offs. They monitor index health, compact "
     "fragments, and tune nprobes and refine factors to keep tail latency tight under bursty "
     "production query workloads."),
]

UNRELATED_TOPICS = {
    1: ("This region's cuisine emphasizes slow-roasted root vegetables and fermented sauces. "
        "A typical Sunday meal pairs hearty bread with stewed legumes."),
    2: ("The northern trail crosses a granite ridge dotted with stunted alpine pines. Hikers "
        "should carry layered clothing because afternoon winds can be biting."),
    3: ("In the late medieval period, illuminated manuscripts depicted seasonal labors with "
        "remarkable detail. Scribes worked in chilled scriptoria from dawn to dusk."),
    4: ("Counterpoint exercises in the Baroque tradition required composers to weave multiple "
        "independent melodic lines without violating species rules of voice leading."),
    5: ("Heritage tomato varieties grown in raised beds benefit from companion planting with "
        "basil and marigold to discourage common garden pests."),
    6: ("Amateur astronomers tracking Jupiter's moons use small refractor telescopes and "
        "logbooks to record transit timings across consecutive observing nights."),
    7: ("Endgame technique in chess often hinges on opposition, triangulation, and accurate "
        "calculation of king and pawn races toward promotion squares."),
    8: ("The poet's sonnet sequence explores grief through repeating tidal imagery, building "
        "from quiet sadness to a final defiant volta."),
    9: ("Hand-cut dovetail joints in cabinet making demand a sharp marking knife, a fine "
        "tenon saw, and patient paring with a well-honed bench chisel."),
}

TOPIC_0_FILLER = ("A short note about an unrelated technology fair held last spring at the "
                  "town hall. The exhibit featured handmade clocks and several restored "
                  "gramophones from the early twentieth century.")


def build_chunk(topic_id: int, idx: int) -> str:
    if topic_id == 0 and idx < 5:
        base = RELEVANT_VDB[idx]
        target = 170 + idx * 4  # 170, 174, 178, 182, 186
        return pad_to_tokens(
            base, target,
            "Operators tune nprobes, refine factors, and quantization for production retrieval.",
        )
    if topic_id == 0:
        target = 60 + (idx % 4) * 30  # 60, 90, 120, 150
        return pad_to_tokens(
            TOPIC_0_FILLER, target,
            "Visitors enjoyed cider and listened to a string quartet on the lawn.",
        )
    base = UNRELATED_TOPICS[topic_id]
    target = 50 + ((idx * 17 + topic_id) % 8) * 25  # 50..225
    return pad_to_tokens(
        base + f" Note {topic_id}-{idx}.",
        target,
        f"Additional notes for topic {topic_id} continue with mundane observations.",
    )


chunks = []
for t in range(10):
    for i in range(15):
        cid = t * 15 + i
        text = build_chunk(t, i)
        tok = len(ENC.encode(text))
        chunks.append({"id": cid, "text": text, "topic_id": t, "tokens": tok})
assert len(chunks) == 150

client = OpenAI()

# Batch-embed in chunks of 100.
all_vecs = []
for s in range(0, len(chunks), 100):
    batch_texts = [c["text"] for c in chunks[s:s + 100]]
    resp = client.embeddings.create(model=MODEL, input=batch_texts)
    all_vecs.extend([d.embedding for d in resp.data])
assert len(all_vecs) == 150

# Sanity-check the rig.
query = "How do vector databases handle production retrieval pipelines?"
qresp = client.embeddings.create(model=MODEL, input=query)
qvec = np.asarray(qresp.data[0].embedding, dtype=np.float32)
mat = np.asarray(all_vecs, dtype=np.float32)
qn = qvec / np.linalg.norm(qvec)
mn = mat / np.linalg.norm(mat, axis=1, keepdims=True)
dists = 1.0 - mn @ qn
order = np.argsort(dists)

top5 = order[:5].tolist()
top5_topics = [chunks[i]["topic_id"] for i in top5]
print("Top-5 ids:", top5, "topics:", top5_topics)
assert top5_topics.count(0) >= 3, (
    f"Diversity rig failed: top-5 topics = {top5_topics}"
)


def pack(order_list, budget, diverse):
    sel, tc, tot = [], {}, 0
    for i in order_list:
        toks = chunks[i]["tokens"]
        if tot + toks > budget:
            continue
        if diverse and tc.get(chunks[i]["topic_id"], 0) >= 2:
            continue
        sel.append(i)
        tot += toks
        tc[chunks[i]["topic_id"]] = tc.get(chunks[i]["topic_id"], 0) + 1
    return sel, tot


order_list = order.tolist()
sel_d, tot_d = pack(order_list, 600, True)
sel_n, tot_n = pack(order_list, 600, False)
print(f"Diverse pack: ids={sel_d} total_tokens={tot_d}")
print(f"Naive pack:   ids={sel_n} total_tokens={tot_n}")
assert sel_d != sel_n, (
    "Rig invariant failed: diversity rule did not change packing at budget 600."
)

# Write to LanceDB.
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("topic_id", pa.int64()),
    pa.field("tokens", pa.int64()),
    pa.field("embedding", pa.list_(pa.float32(), 1536)),
])
rows = []
for c, v in zip(chunks, all_vecs):
    rows.append({**c, "embedding": [float(x) for x in v]})

db = lancedb.connect(LANCEDB_DIR)
table_name = f"chunks_{RUN_ID}"
if table_name in db.table_names():
    db.drop_table(table_name)
tbl = db.create_table(table_name, data=rows, schema=schema, mode="overwrite")
print(f"Seeded {tbl.count_rows()} rows into '{table_name}'.")
# lancedb 0.25.3 occasionally hits a SIGABRT during interpreter teardown; bail cleanly.
os._exit(0)
PYEOF

# Entrypoint script: idempotently seed once per ZEALT_RUN_ID, then idle.
RUN cat >/app/entrypoint.sh <<'EOF'
#!/bin/bash
set -e

if [ -z "${ZEALT_RUN_ID}" ]; then
    echo "ERROR: ZEALT_RUN_ID is not set." >&2
    exit 1
fi
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi

MARKER="/app/.seeded_${ZEALT_RUN_ID}"
if [ ! -f "${MARKER}" ]; then
    echo "[entrypoint] Seeding LanceDB table chunks_${ZEALT_RUN_ID}..."
    python3 /app/_seed.py
    touch "${MARKER}"
else
    echo "[entrypoint] Seed marker present at ${MARKER}; skipping seed."
fi

exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
EOF
RUN chmod +x /app/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB token-budget context-packing task: 150 chunks embedded with real OpenAI text-embedding-3-small at entrypoint, rigged so the topic-diversity rule triggers at max_tokens=600."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#          openai==1.54.5 tiktoken httpx==0.27.2
#   - /home/user/myproject: candidate solution directory
#   - /app/lancedb_data: LanceDB local store
#   - /app/_seed.py: seed script (runs at entrypoint, embeds 150 chunks via real OpenAI)
#   - /app/entrypoint.sh: idempotent entrypoint, seeds once per ZEALT_RUN_ID
