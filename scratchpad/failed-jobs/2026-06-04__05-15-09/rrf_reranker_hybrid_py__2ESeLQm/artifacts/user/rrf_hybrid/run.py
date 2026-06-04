"""
Hybrid Vector + FTS Search with RRF Reranker using LanceDB.

Connects to LanceDB, creates a deterministic table `kb`, builds a native FTS
index, runs a hybrid search fused with RRFReranker, and writes the top-5
results to /workspace/output/hybrid_rrf.json.
"""

import json
import os

import numpy as np
import pyarrow as pa
import lancedb
from lancedb.rerankers import RRFReranker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LANCEDB_URI = os.environ.get("LANCEDB_URI", "/workspace/db")
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hybrid_rrf.json")
TABLE_NAME = "kb"

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
TEXTS = [
    "introduction to REDACTEDs",           #  0
    "columnar storage in arrow",                  #  1
    "lancedb columnar storage backend",           #  2
    "approximate nearest neighbor algorithms",    #  3
    "fts and bm25 keyword search",                #  4
    "vector search benchmark suite",              #  5
    "embedding models for retrieval",             #  6
    "hybrid rrf reranker tutorial for lancedb",   #  7
    "ivf pq quantization basics",                 #  8
    "hnsw graph index overview",                  #  9
    "cosine similarity vs dot product",           # 10
    "euclidean distance considerations",          # 11
    "tantivy full text search engine",            # 12
    "native lance fts native indexing",           # 13
    "metadata filtering with sql",                # 14
    "pyarrow integration patterns",               # 15
    "open source ai retrieval stack",             # 16
    "rag pipelines with langchain",               # 17
    "llamaindex vector store overview",           # 18
    "multimodal embeddings for images",           # 19
    "blob storage in lance datasets",             # 20
    "merge insert upsert workflows",              # 21
    "table versioning and time travel",           # 22
    "schema evolution with arrow",                # 23
    "delete by predicate semantics",              # 24
    "update with sql where clauses",              # 25
    "automatic index compaction strategies",      # 26
    "s3 minio cloud deployment",                  # 27
    "embedding registry openai integration",      # 28
    "deterministic seeding for tests",            # 29
]

# ---------------------------------------------------------------------------
# Deterministic vectors: seed=7, shape=(30, 8), float32
# ---------------------------------------------------------------------------
VECTORS = np.random.default_rng(7).standard_normal((30, 8)).astype("float32")

# ---------------------------------------------------------------------------
# Arrow schema  (column order: id, text, vector)
# ---------------------------------------------------------------------------
SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 8)),
])

# ---------------------------------------------------------------------------
# Build the table data as a PyArrow Table
# ---------------------------------------------------------------------------
ids = list(range(30))
vectors_list = VECTORS.tolist()          # list of 30 lists of 8 floats

arrow_table = pa.table(
    {
        "id": pa.array(ids, type=pa.int64()),
        "text": pa.array(TEXTS, type=pa.string()),
        "vector": pa.array(vectors_list, type=pa.list_(pa.float32(), 8)),
    },
    schema=SCHEMA,
)

# ---------------------------------------------------------------------------
# Connect and (re)create the table
# ---------------------------------------------------------------------------
print(f"Connecting to LanceDB at: {LANCEDB_URI}")
db = lancedb.connect(LANCEDB_URI)

print(f"Creating table '{TABLE_NAME}' (mode=overwrite) …")
table = db.create_table(TABLE_NAME, data=arrow_table, schema=SCHEMA, mode="overwrite")
print(f"  → {table.count_rows()} rows written.")

# ---------------------------------------------------------------------------
# Build native FTS index (Tantivy disabled, replace any existing index)
# ---------------------------------------------------------------------------
print("Building native FTS index on 'text' column …")
table.create_fts_index("text", use_tantivy=False, replace=True)
print("  → FTS index created.")

# Confirm index list
indices = table.list_indices()
print(f"  → Indices reported: {indices}")

# ---------------------------------------------------------------------------
# Hybrid search: vector similarity + FTS, fused with RRF
# ---------------------------------------------------------------------------
query_vec = VECTORS[7]          # row 7 of the seeded dataset (deterministic)
query_text = "rrf reranker"

print(f"\nRunning hybrid search …")
print(f"  query_text  : {query_text!r}")
print(f"  query_vec   : row 7 → {query_vec}")

results = (
    table.search(query_type="hybrid")
    .vector(query_vec)
    .text(query_text)
    .rerank(RRFReranker())
    .limit(5)
    .to_list()
)

print(f"\nTop-5 results:")
for rank, row in enumerate(results, 1):
    print(f"  {rank}. id={row['id']}  text={row['text']!r}")

# ---------------------------------------------------------------------------
# Serialise results → [id, text] pairs and write JSON
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

output = [[int(row["id"]), row["text"]] for row in results]

with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
    json.dump(output, fh, ensure_ascii=False, indent=2)

print(f"\nOutput written to: {OUTPUT_FILE}")
print(json.dumps(output, indent=2))
