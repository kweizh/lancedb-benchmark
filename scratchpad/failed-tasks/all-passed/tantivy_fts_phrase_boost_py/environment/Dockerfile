FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pylance==0.21.0 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        tantivy==0.22.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

RUN mkdir -p /home/user/myproject/seed

# Build-time seed generator: writes a deterministic 60-doc corpus to
# /home/user/myproject/seed/docs.json. The corpus is rigged so that exactly:
#   * doc id=57 contains the contiguous phrase "vector database" in its body
#     (and "Vector Database" in its title).
#   * doc id=58 is the only doc with "lancedb" in its title.
#   * doc id=59 is the only doc whose body contains BOTH "rust" and "tantivy".
# All other docs draw deterministically from a controlled token pool that
# excludes the rigged tokens, so the verifier's queries have a unique winner.
RUN cat >/tmp/build_seed.py <<'EOF'
import json
import os
import random

ADJECTIVES = [
    "fast", "modern", "robust", "scalable", "modular",
    "lazy", "eager", "compact", "verbose", "stateful",
    "stateless", "atomic", "consistent", "durable", "isolated",
]
NOUNS = [
    "search", "engine", "graph", "matrix", "tensor",
    "embedding", "neural", "cluster", "shard", "kernel",
    "tree", "scalar", "binary", "transformer", "optimizer",
    "gradient", "attention", "pipeline", "schema", "storage",
    "checkpoint", "service", "metric", "ranker", "encoder",
    "decoder", "buffer", "queue", "thread", "session",
]
VERBS = [
    "accelerates", "enables", "compresses", "optimizes",
    "benchmarks", "indexes", "shards", "rebalances",
    "filters", "ranks", "scores", "evaluates",
    "compiles", "serializes",
]

# Tokens that are reserved for the rigged docs only; we sanity-check they
# never appear in any non-rigged title/body.
RESERVED = {"vector", "database", "lancedb", "rust", "tantivy"}

rng = random.Random(2026)
docs = []
for i in range(57):
    a1 = rng.choice(ADJECTIVES)
    n1 = rng.choice(NOUNS)
    n2 = rng.choice(NOUNS)
    title = f"{a1.capitalize()} {n1.capitalize()} {n2.capitalize()}"
    v = rng.choice(VERBS)
    n3 = rng.choice(NOUNS)
    a2 = rng.choice(ADJECTIVES)
    n4 = rng.choice(NOUNS)
    body = f"The {n1} {v} the {n3} with {a2} {n4} data."
    docs.append({"id": i, "title": title, "body": body})

# Rigged winners.
docs.append({
    "id": 57,
    "title": "Modern Vector Database Systems",
    "body": (
        "A vector database accelerates similarity search for retrieval "
        "augmented generation workloads."
    ),
})
docs.append({
    "id": 58,
    "title": "LanceDB Tutorial Guide",
    "body": (
        "This tutorial explores scalar and vector field configuration for "
        "hybrid retrieval pipelines."
    ),
})
docs.append({
    "id": 59,
    "title": "Native Search Performance Notes",
    "body": (
        "The rust runtime and tantivy library power native full text "
        "search in modern engines."
    ),
})

# Integrity assertions: rigged tokens never appear in non-rigged docs.
for doc in docs[:57]:
    blob = (doc["title"] + " " + doc["body"]).lower()
    for tok in RESERVED:
        assert tok not in blob, (
            f"Reserved token '{tok}' leaked into non-rigged doc {doc['id']}: {blob}"
        )

# Phrase uniqueness assertion: only doc 57 contains the contiguous phrase.
phrase = "vector database"
phrase_hits = [d for d in docs if phrase in d["body"].lower()]
assert len(phrase_hits) == 1 and phrase_hits[0]["id"] == 57

# lancedb-in-title uniqueness assertion.
lancedb_title_hits = [d for d in docs if "lancedb" in d["title"].lower()]
assert len(lancedb_title_hits) == 1 and lancedb_title_hits[0]["id"] == 58

# rust+tantivy-in-body uniqueness assertion.
both_hits = [
    d for d in docs
    if "rust" in d["body"].lower() and "tantivy" in d["body"].lower()
]
assert len(both_hits) == 1 and both_hits[0]["id"] == 59

out_path = "/home/user/myproject/seed/docs.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2)
print(f"Wrote {len(docs)} docs to {out_path}")
EOF
RUN python3 /tmp/build_seed.py && rm /tmp/build_seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB Tantivy-backed FTS task with a rigged 60-doc corpus for phrase, field-boost, and boolean-AND queries."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, tantivy==0.22.0
#     for the LanceDB Tantivy-backed FTS task and verifier
#   - /home/user/myproject: project root
#   - /home/user/myproject/seed/docs.json: deterministic 60-doc corpus with
#     rigged winners for phrase/field-boost/boolean-AND queries
