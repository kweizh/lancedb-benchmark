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
        httpx==0.27.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Bake the seed script into the image.
RUN cat >/home/user/myproject/_seed.py <<'PYEOF'
"""Seed `/home/user/myproject/lancedb` with a 200-row `docs` table.

* Two sentinel rows: `rigged-correct` (the real answer) and `rigged-distractor`
  (a lexically identical paraphrase of the anchor query that does NOT answer it).
* 198 unrelated filler rows.

All 200 contents are embedded once with real OpenAI `text-embedding-3-small`
using the build-time OPENAI_API_KEY ARG, then written into a LanceDB table
whose schema is `(id: string, content: string, embedding: fixed_size_list<float, 1536>)`.

Before writing the table, the script computes the cosine similarity between
the anchor query and every doc and asserts the **fixture invariant** that the
distractor ranks strictly above the correct answer in pure vector search. This
is what makes the cross-encoder rerank step necessary at evaluation time.
"""

import json
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/home/user/myproject/lancedb"
EXPECTED_JSON = "/home/user/myproject/.expected.json"
TABLE_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

ANCHOR_QUERY = "what command undoes the most recent git commit"

# The actual answer. Paraphrased so it does NOT verbatim contain the query string.
RIGGED_CORRECT = {
    "id": "rigged-correct",
    "content": (
        "To rewind your branch pointer by one revision while leaving your working "
        "tree untouched, run: git reset --soft HEAD~1. If instead you want to discard "
        "those changes entirely, use git reset --hard HEAD~1. Either form moves the "
        "branch back one revision."
    ),
}

# Verbatim restatement of the query, repeated several times, followed by an
# explicit statement that it does NOT contain the answer. This maximises raw
# embedding overlap with the query string while a competent cross-encoder will
# still recognise that no answer is provided.
RIGGED_DISTRACTOR = {
    "id": "rigged-distractor",
    "content": (
        "What command undoes the most recent git commit. "
        "What command undoes the most recent git commit. "
        "What command undoes the most recent git commit. "
        "This entry is a placeholder section title and does not contain any answer, "
        "instruction, or command. See the FAQ for actual guidance."
    ),
}

# 198 unrelated short filler topics. Repeat across variants to reach 198.
FILLER_TEMPLATES = [
    "A classic Italian tomato pasta recipe with fresh basil and garlic, perfect for dinner. Variant {n}.",
    "Quarterly earnings report highlights record revenue growth in the cloud division. Variant {n}.",
    "Travel guide to Kyoto: cherry blossoms, temples, and tea ceremonies in spring. Variant {n}.",
    "Beginners guide to home gardening: how to grow tomatoes and herbs on a balcony. Variant {n}.",
    "Stock market commentary: tech sector rebounds after a long correction. Variant {n}.",
    "Tips for training a puppy to sit, stay, and walk on a leash without pulling. Variant {n}.",
    "Photography composition basics: rule of thirds, leading lines, and negative space. Variant {n}.",
    "Beginner yoga routine for office workers with tight hips and stiff shoulders. Variant {n}.",
    "Watercolor painting techniques for landscapes: wet-on-wet washes and dry brush details. Variant {n}.",
    "How to brew a perfect pour-over coffee using a Hario V60 and a goose-neck kettle. Variant {n}.",
    "Marathon training plan covering long runs, tempo workouts, and rest week recovery. Variant {n}.",
    "Astronomy basics: identifying planets and constellations in the northern sky. Variant {n}.",
    "Cooking class focused on classic French sauces: bechamel, veloute, and hollandaise. Variant {n}.",
    "Personal finance basics: how to build an emergency fund and pay off credit card debt. Variant {n}.",
    "Beginners electric guitar lesson: power chords, palm muting, and basic blues licks. Variant {n}.",
    "Sourdough bread starter maintenance: feeding schedule, hydration, and discard recipes. Variant {n}.",
    "Hiking the Pacific Crest Trail: gear list, water sources, and resupply strategy. Variant {n}.",
    "Birdwatching journal entry: a morning of warblers in the spring migration. Variant {n}.",
    "Pottery wheel throwing tips: centering clay, opening, and pulling even walls. Variant {n}.",
    "Cycling commute tips: pannier setup, rain gear, and visible night lighting. Variant {n}.",
    "Beekeeping basics: hive inspection schedule, smoker technique, and varroa monitoring. Variant {n}.",
    "Knitting a basic ribbed beanie: cast-on count, ribbing pattern, and decrease rounds. Variant {n}.",
    "Camping in the desert: hydration strategy, sun shelter, and cold-night layering. Variant {n}.",
    "Ocean kayaking safety: tide tables, paddle float self-rescue, and VHF radio basics. Variant {n}.",
    "Foraging mushrooms responsibly: spore prints, look-alikes, and safe identification keys. Variant {n}.",
    "Bonsai care guide: pruning windows, root pruning frequency, and winter protection. Variant {n}.",
    "Brewing English-style cask ale: gravity targets, dry hopping, and conditioning time. Variant {n}.",
    "Restoring an old film camera: lens fungus removal, shutter timing, and meter recalibration. Variant {n}.",
    "Beginning origami: valley folds, mountain folds, and a simple flapping crane. Variant {n}.",
    "Maintaining a saltwater aquarium: salinity, alkalinity, and protein skimmer tuning. Variant {n}.",
    "Choosing running shoes: heel-to-toe drop, midsole foam, and gait analysis tips. Variant {n}.",
    "Identifying tree species by leaf shape, bark texture, and twig branching pattern. Variant {n}.",
    "Home roasting coffee in a popcorn popper: first crack timing and rest period. Variant {n}.",
    "Hand-lettering basics: stroke contrast, baseline consistency, and flourish placement. Variant {n}.",
    "Tying common fishing knots: improved clinch, palomar, and double surgeons loop. Variant {n}.",
    "Building a small backyard pond: liner choice, edging stones, and aeration. Variant {n}.",
    "Stretching routine for cyclists: hip flexors, hamstrings, and lower back release. Variant {n}.",
    "Letterpress printing introduction: lockup, ink modulation, and impression depth. Variant {n}.",
    "Choosing climbing rope: dry treatment, fall rating, and sheath construction. Variant {n}.",
    "Caring for a cast-iron skillet: seasoning, scrubbing, and rust recovery. Variant {n}.",
    "Beginner snowboarding: heel-side carving, toe-side carving, and basic turns. Variant {n}.",
    "Starting a vegetable garden indoors: grow lights, seedling trays, and hardening off. Variant {n}.",
    "Beginning bouldering: footwork drills, body positioning, and rest beta. Variant {n}.",
    "Brewing kombucha at home: SCOBY care, second fermentation, and flavor pairings. Variant {n}.",
    "Choosing a beginner mountain bike: hardtail versus full suspension and wheel size. Variant {n}.",
    "Caring for indoor succulents: watering rhythm, soil mix, and pup separation. Variant {n}.",
    "Mountain photography in winter: snow exposure compensation and battery warming. Variant {n}.",
    "Beginner sea fishing from shore: rig setup, surf timing, and tide reading. Variant {n}.",
    "Whittling a wooden spoon: blank selection, bowl carving, and finishing oil. Variant {n}.",
    "Restoring an old wooden chair: glue joints, dowel replacement, and refinishing. Variant {n}.",
]


def build_rows():
    rows = [RIGGED_CORRECT.copy(), RIGGED_DISTRACTOR.copy()]
    filler_count = 0
    n = 1
    while filler_count < 198:
        for i, tpl in enumerate(FILLER_TEMPLATES):
            if filler_count >= 198:
                break
            rows.append({
                "id": f"filler-{i:02d}-{n}",
                "content": tpl.format(n=n),
            })
            filler_count += 1
        n += 1
    assert len(rows) == 200, len(rows)
    # All ids must be unique.
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "Duplicate ids in seed rows!"
    return rows


def embed_batch(client, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for item in resp.data:
            out.append(item.embedding)
    return out


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("[seed] ERROR: OPENAI_API_KEY is required at build time.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DB_DIR, exist_ok=True)

    rows = build_rows()
    texts = [r["content"] for r in rows]

    client = OpenAI()
    print(f"[seed] Embedding {len(texts)} documents with {EMBED_MODEL}...", flush=True)
    embeddings = embed_batch(client, texts)
    assert len(embeddings) == len(rows), (len(embeddings), len(rows))
    for r, vec in zip(rows, embeddings):
        assert len(vec) == EMBED_DIM, (r["id"], len(vec))
        r["embedding"] = vec

    # Embed the anchor query independently and verify the fixture invariant.
    qresp = client.embeddings.create(model=EMBED_MODEL, input=[ANCHOR_QUERY])
    qvec = qresp.data[0].embedding

    id_to_emb = {r["id"]: r["embedding"] for r in rows}
    sim_correct = _cosine(qvec, id_to_emb["rigged-correct"])
    sim_distractor = _cosine(qvec, id_to_emb["rigged-distractor"])
    print(f"[seed] cos(query, rigged-correct)={sim_correct:.4f}", flush=True)
    print(f"[seed] cos(query, rigged-distractor)={sim_distractor:.4f}", flush=True)
    assert sim_distractor > sim_correct, (
        "Fixture invariant failed at build time: vector-only similarity should put the "
        f"distractor above the correct doc. Got distractor={sim_distractor:.4f}, "
        f"correct={sim_correct:.4f}. Adjust the rigged content."
    )

    # Confirm both sentinels are actually in the vector-only top-30.
    sims = sorted(
        ((rid, _cosine(qvec, emb)) for rid, emb in id_to_emb.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top30_ids = [rid for rid, _ in sims[:30]]
    assert "rigged-correct" in top30_ids, top30_ids
    assert "rigged-distractor" in top30_ids, top30_ids
    rank_d = top30_ids.index("rigged-distractor")
    rank_c = top30_ids.index("rigged-correct")
    print(f"[seed] vector-only top-30 rank(distractor)={rank_d} rank(correct)={rank_c}", flush=True)
    assert rank_d < rank_c, (rank_d, rank_c)

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
    ])
    arrays = {
        "id": [r["id"] for r in rows],
        "content": [r["content"] for r in rows],
        "embedding": [np.asarray(r["embedding"], dtype=np.float32) for r in rows],
    }
    table_arrow = pa.table(arrays, schema=schema)

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=table_arrow, mode="create")
    n_rows = tbl.count_rows()
    print(f"[seed] Table '{TABLE_NAME}' now has {n_rows} rows.", flush=True)
    assert n_rows == 200, n_rows

    with open(EXPECTED_JSON, "w") as f:
        json.dump(
            {
                "anchor_query": ANCHOR_QUERY,
                "rigged_correct_id": "rigged-correct",
                "rigged_distractor_id": "rigged-distractor",
                "vector_rank_correct": rank_c,
                "vector_rank_distractor": rank_d,
                "cos_correct": sim_correct,
                "cos_distractor": sim_distractor,
            },
            f,
            indent=2,
        )
    print(f"[seed] Wrote {EXPECTED_JSON}.", flush=True)
    # Avoid lancedb 0.25.3 SIGABRT at interpreter teardown after heavy writes.
    os._exit(0)


main()
PYEOF

# Build-time seed: requires --build-arg OPENAI_API_KEY=...
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
RUN test -n "$OPENAI_API_KEY" && \
    python3 /home/user/myproject/_seed.py
# Scrub the key out of the image layer; runtime will inject a fresh value.
ENV OPENAI_API_KEY=

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB + HF Inference cross-encoder rerank task: 200-row docs table seeded at build time with real OpenAI text-embedding-3-small (1536-d). Two sentinel rows are rigged so vector-only top-1 is a keyword-stuffed distractor; only a working cross-encoder rerank step against the Hugging Face Inference API can recover the right answer at rank 1."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3 + pyarrow==17.0.0 + numpy==1.26.4 + pandas==2.2.3 + openai==1.54.5 + httpx==0.27.2
#   - /home/user/myproject/_seed.py: build-time seed script (already executed during image build)
#   - /home/user/myproject/lancedb/docs: 200-row LanceDB table with 1536-d real OpenAI embeddings
#   - /home/user/myproject/.expected.json: anchor query + rigged ids fixture for the verifier
