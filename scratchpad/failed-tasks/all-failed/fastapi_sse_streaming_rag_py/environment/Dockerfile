FROM ubuntu:24.04

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "fastapi==0.115.5" \
        "uvicorn[standard]==0.32.1" \
        "sse-starlette==2.1.3" \
        "lancedb==0.25.3" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pandas==2.2.3" \
        "openai==1.54.5" \
        "httpx==0.27.2" \
        "tiktoken" \
        "pytest-xprocess" \
        "requests" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject /home/user/myproject/data

# Inline build-time seed script: embeds ~30 doc chunks with real OpenAI
# text-embedding-3-small into a LanceDB table and pre-computes the expected
# top-3 retrieval for the fixture question.
RUN cat > /tmp/seed.py <<'PY'
import json
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/home/user/myproject/data/lancedb"
FIXTURE_PATH = "/home/user/myproject/.expected.json"
TABLE_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

DOCS = [
    ("doc_pydantic_model",
     "In LanceDB you can define a typed table schema by subclassing lancedb.pydantic.LanceModel. "
     "Each Pydantic field becomes a column; use Vector(ndims) for the embedding column. "
     "LanceModel makes the table schema discoverable and validated by Pydantic."),
    ("doc_pydantic_vector_field",
     "Use lancedb.pydantic.Vector to declare a fixed-size float vector field inside a LanceModel. "
     "Vector(1536) marks the column as a 1536-dim embedding. You can also annotate the source text "
     "column with func.SourceField() and the vector column with func.VectorField() when wiring an "
     "embedding registry to a Pydantic schema."),
    ("doc_pydantic_typed_schema",
     "A typed LanceDB schema built with Pydantic LanceModel lets you create a table from a Python "
     "class instead of a raw pyarrow.Schema. db.create_table(name, schema=MyLanceModel) infers the "
     "Arrow schema from the Pydantic model and validates incoming rows."),

    ("doc_vector_search",
     "Run a vector similarity search on a table with table.search(query_vector).limit(k).to_list() "
     "to retrieve the k nearest neighbours of a query embedding."),
    ("doc_create_table",
     "Create or overwrite a LanceDB table with db.create_table(name, data=rows, mode='overwrite'). "
     "Open an existing table with db.open_table(name)."),
    ("doc_ivf_pq",
     "Build an IVF_PQ vector index with table.create_index(index_type='IVF_PQ', metric='cosine', "
     "num_partitions=N, num_sub_vectors=M). Training requires at least 256 rows."),
    ("doc_sql_filter",
     "Use SQL-style metadata filters with table.search(vec).where(\"category = 'a' AND price < 10\") "
     "to combine vector search with structured predicates."),
    ("doc_hybrid_search",
     "Hybrid search combines vector and full-text scoring: "
     "table.search(query_type='hybrid').vector(vec).text(text).rerank(RRFReranker()).limit(k)."),
    ("doc_fts_index",
     "Native full-text search uses table.create_fts_index('content', use_tantivy=False) followed by "
     "table.search(query, query_type='fts')."),
    ("doc_versioning",
     "LanceDB tables are versioned. Inspect history with table.list_versions(), pin a snapshot with "
     "table.checkout(version), and return to head with table.checkout_latest()."),
    ("doc_embedding_registry",
     "Register an embedding function via get_registry().get('openai').create(name='text-embedding-3-small') "
     "and bind it to a Pydantic field with func.VectorField() so queries auto-embed plain strings."),
    ("doc_merge_insert",
     "Upsert rows with table.merge_insert('id').when_matched_update_all().when_not_matched_insert_all().execute(rows)."),
    ("doc_delete",
     "Delete rows matching a SQL predicate with table.delete(\"id IN (1,2,3)\")."),
    ("doc_add_columns",
     "Schema evolution: add columns derived from SQL expressions with "
     "table.add_columns({'cents': 'CAST(price*100 AS BIGINT)'})."),
    ("doc_update",
     "Update column values in place with table.update(where=\"id = '42'\", values={'name': \"O'Brien\"})."),
    ("doc_async",
     "lancedb.connect_async returns an AsyncConnection whose AsyncTable methods can be awaited "
     "and fanned out with asyncio.gather across many queries."),
    ("doc_s3",
     "Connect to an S3-backed store with lancedb.connect('s3://bucket/path', storage_options={...})."),
    ("doc_record_batch",
     "Stream large datasets via pyarrow.RecordBatchReader.from_batches and pass the reader to "
     "create_table for batched ingestion."),
    ("doc_rrf",
     "RRFReranker fuses vector and FTS hit lists using reciprocal rank fusion; use it as the "
     "rerank step in a hybrid search builder."),
    ("doc_distance",
     "Pick a distance metric per query with table.search(vec).distance_type('L2' | 'cosine' | 'dot')."),
    ("doc_optimize",
     "table.optimize(cleanup_older_than=timedelta(days=7)) compacts fragments and prunes old "
     "table versions to reclaim disk space."),
    ("doc_index_stats",
     "table.index_stats('<index_name>') returns metadata including num_unindexed_rows, which is "
     "useful for driving automatic re-indexing loops."),
    ("doc_multimodal",
     "Store binary blobs alongside vectors using pa.large_binary() columns; tables can mix image "
     "bytes, text, and float vectors in a single row."),
    ("doc_batch_search",
     "table.search() accepts a 2-D numpy array of query vectors for batched nearest-neighbour "
     "search; the returned DataFrame includes a query_index column."),
    ("doc_langchain",
     "langchain_community.vectorstores.LanceDB wraps a LanceDB table as a LangChain vector store "
     "compatible with OpenAIEmbeddings and retriever chains."),
    ("doc_llamaindex",
     "llama_index.vector_stores.lancedb.LanceDBVectorStore plugs LanceDB into LlamaIndex; metadata "
     "filters are pushed down to the underlying SQL where clause."),
    ("doc_pagination",
     "Page through results deterministically by combining .limit(page_size) with .offset((page-1)*page_size)."),
    ("doc_btree",
     "table.create_scalar_index('price', index_type='BTREE') accelerates range predicates on "
     "numeric columns."),
    ("doc_bitmap",
     "BITMAP scalar indexes are best for low-cardinality columns; create one with "
     "table.create_scalar_index('tag', index_type='BITMAP') and combine with prefilter=True for "
     "vector queries."),
    ("doc_count_rows",
     "Count matching rows without materializing them via table.count_rows(filter=\"category = 'a'\")."),
]

def embed_batch(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]

def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set at build time.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    ids = [d[0] for d in DOCS]
    contents = [d[1] for d in DOCS]

    # Embed docs in one batch.
    doc_vecs = embed_batch(client, contents)
    assert all(v.shape == (EMBED_DIM,) for v in doc_vecs), "Unexpected embedding dimension"

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ])
    rows = [
        {"id": ids[i], "content": contents[i], "vector": doc_vecs[i].tolist()}
        for i in range(len(ids))
    ]

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    tbl = db.create_table(TABLE_NAME, data=rows, schema=schema)
    assert tbl.count_rows() == len(rows), "Seed row count mismatch"

    # Pick the fixture question. It should map strongly to the Pydantic/LanceModel cluster.
    question = "How do I define a typed LanceDB table schema using Pydantic LanceModel and a Vector field?"

    # Embed the question and compute the top-3 retrieval that the candidate
    # service must reproduce at evaluation time.
    qvec = embed_batch(client, [question])[0]
    hits = tbl.search(qvec.tolist()).limit(3).to_list()
    expected_sources = [h["id"] for h in hits]
    assert len(expected_sources) == 3, "Expected exactly 3 retrieval hits"
    print("[seed] expected_sources:", expected_sources)

    # Sanity check that the top-3 has a margin over the 4th hit, otherwise the
    # retrieval may be unstable across repeated OpenAI calls.
    top4 = tbl.search(qvec.tolist()).limit(4).to_list()
    if len(top4) == 4:
        d3 = top4[2].get("_distance", 0.0)
        d4 = top4[3].get("_distance", 0.0)
        margin = float(d4) - float(d3)
        print(f"[seed] margin between hit3 and hit4: {margin:.6f}")

    fixture = {
        "question": question,
        "expected_sources": expected_sources,
    }
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print("[seed] wrote fixture to", FIXTURE_PATH)

    # Work around lancedb 0.25.3 occasional SIGABRT during interpreter shutdown
    # in write-heavy build-time seed scripts.
    os._exit(0)

main()
PY

RUN python3 /tmp/seed.py && rm -f /tmp/seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="FastAPI SSE streaming RAG over a pre-seeded LanceDB table with real OpenAI embeddings."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, fastapi, uvicorn[standard], sse-starlette, lancedb==0.25.3,
#           pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, openai==1.54.5, tiktoken,
#           pytest-xprocess, requests
#   - /home/user/myproject: candidate project root
#   - /home/user/myproject/data/lancedb: pre-seeded LanceDB store with ~30 docs (real OpenAI embeddings)
#   - /home/user/myproject/.expected.json: build-time fixture {question, expected_sources}
#   - OPENAI_API_KEY: real key consumed at build (seed) and runtime (candidate + verifier)
