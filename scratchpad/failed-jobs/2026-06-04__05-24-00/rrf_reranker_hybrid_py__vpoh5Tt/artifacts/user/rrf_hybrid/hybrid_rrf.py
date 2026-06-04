#!/usr/bin/env python3
"""Hybrid Vector + FTS Search with RRF Reranker on LanceDB."""

import json
import os

import numpy as np
import pyarrow as pa
import lancedb
from lancedb.rerankers import RRFReranker

# ── Configuration ──────────────────────────────────────────────────────────
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
TABLE_NAME = "kb"
OUTPUT_PATH = "/workspace/output/hybrid_rrf.json"

# ── Seed data ──────────────────────────────────────────────────────────────
TEXTS = [
    "introduction to REDACTEDs",         #  0
    "columnar storage in arrow",                 #  1
    "lancedb columnar storage backend",          #  2
    "approximate nearest neighbor algorithms",  #  3
    "fts and bm25 keyword search",              #  4
    "vector search benchmark suite",             #  5
    "embedding models for retrieval",            #  6
    "hybrid rrf reranker tutorial for lancedb", #  7
    "ivf pq quantization basics",               #  8
    "hnsw graph index overview",                #  9
    "cosine similarity vs dot product",         # 10
    "euclidean distance considerations",        # 11
    "tantivy full text search engine",           # 12
    "native lance fts native indexing",          # 13
    "metadata filtering with sql",               # 14
    "pyarrow integration patterns",              # 15
    "open source ai retrieval stack",           # 16
    "rag pipelines with langchain",             # 17
    "llamaindex vector store overview",         # 18
    "multimodal embeddings for images",         # 19
    "blob storage in lance datasets",           # 20
    "merge insert upsert workflows",            # 21
    "table versioning and time travel",         # 22
    "schema evolution with arrow",              # 23
    "delete by predicate semantics",             # 24
    "update with sql where clauses",            # 25
    "automatic index compaction strategies",    # 26
    "s3 minio cloud deployment",                # 27
    "embedding registry openai integration",    # 28
    "deterministic seeding for tests",          # 29
]

NUM_ROWS = len(TEXTS)  # 30
VEC_DIM = 8

# ── Deterministic vectors ─────────────────────────────────────────────────
rng = np.random.default_rng(7)
vectors = rng.standard_normal((NUM_ROWS, VEC_DIM)).astype("float32")

# ── Build Arrow table ─────────────────────────────────────────────────────
ids = pa.array(list(range(NUM_ROWS)), type=pa.int64())
texts = pa.array(TEXTS, type=pa.string())
vec_list = pa.FixedSizeListArray.from_arrays(vectors.flatten(), list_size=VEC_DIM)

schema = pa.schema([
    ("id", pa.int64()),
    ("text", pa.string()),
    ("vector", pa.list_(pa.float32(), VEC_DIM)),
])

table_data = pa.Table.from_arrays([ids, texts, vec_list], schema=schema)

# ── Connect & create / overwrite table ─────────────────────────────────────
db = lancedb.connect(LANCEDB_URI)
db.create_table(TABLE_NAME, data=table_data, mode="overwrite")

# ── Build native FTS index ────────────────────────────────────────────────
tbl = db.open_table(TABLE_NAME)
tbl.create_fts_index("text", use_tantivy=False, replace=True)

# ── Hybrid search with RRF reranker ───────────────────────────────────────
query_vec = vectors[7]  # row 7's vector (deterministic)
reranker = RRFReranker()

results = (
    tbl.search(query_type="hybrid")
    .vector(query_vec)
    .text("rrf reranker")
    .rerank(reranker)
    .limit(5)
    .to_list()
)

# ── Materialise output ─────────────────────────────────────────────────────
output = [[int(r["id"]), r["text"]] for r in results]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print("Output written to", OUTPUT_PATH)
print(json.dumps(output, indent=2))