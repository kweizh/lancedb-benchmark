import os
import json
import numpy as np
import pyarrow as pa
import lancedb
from lancedb.rerankers import RRFReranker

def main():
    # 1. Connect to LanceDB
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    print(f"Connecting to LanceDB at: {db_uri}")
    db = lancedb.connect(db_uri)

    # 2. Define schema
    schema = pa.schema([
        ("id", pa.int64()),
        ("text", pa.string()),
        ("vector", pa.list_(pa.float32(), 8))
    ])

    # 3. Prepare data
    texts = [
        "introduction to REDACTEDs",
        "columnar storage in arrow",
        "lancedb columnar storage backend",
        "approximate nearest neighbor algorithms",
        "fts and bm25 keyword search",
        "vector search benchmark suite",
        "embedding models for retrieval",
        "hybrid rrf reranker tutorial for lancedb",
        "ivf pq quantization basics",
        "hnsw graph index overview",
        "cosine similarity vs dot product",
        "euclidean distance considerations",
        "tantivy full text search engine",
        "native lance fts native indexing",
        "metadata filtering with sql",
        "pyarrow integration patterns",
        "open source ai retrieval stack",
        "rag pipelines with langchain",
        "llamaindex vector store overview",
        "multimodal embeddings for images",
        "blob storage in lance datasets",
        "merge insert upsert workflows",
        "table versioning and time travel",
        "schema evolution with arrow",
        "delete by predicate semantics",
        "update with sql where clauses",
        "automatic index compaction strategies",
        "s3 minio cloud deployment",
        "embedding registry openai integration",
        "deterministic seeding for tests"
    ]

    # Seed vectors deterministically
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((30, 8)).astype("float32")

    ids_array = pa.array(range(30), type=pa.int64())
    texts_array = pa.array(texts, type=pa.string())
    vectors_array = pa.array([v.tolist() for v in vectors], type=pa.list_(pa.float32(), 8))

    batch = pa.RecordBatch.from_arrays(
        [ids_array, texts_array, vectors_array],
        schema=schema
    )
    table_data = pa.Table.from_batches([batch])

    # 4. Create or recreate table
    table_name = "kb"
    print(f"Creating table '{table_name}' with mode='overwrite'...")
    table = db.create_table(table_name, data=table_data, schema=schema, mode="overwrite")

    # 5. Build native Lance FTS index on the 'text' column
    print("Building FTS index...")
    table.create_fts_index("text", use_tantivy=False, replace=True)

    # 6. Run hybrid search
    print("Running hybrid search...")
    query_vec = vectors[7].tolist() # Row 7's vector
    results = table.search(query_type="hybrid") \
                   .vector(query_vec) \
                   .text("rrf reranker") \
                   .rerank(RRFReranker()) \
                   .limit(5) \
                   .to_list()

    print("Search results:")
    for r in results:
        print(r)

    # 7. Format top 5 results as JSON [[id, text], ...]
    output_data = [[int(r["id"]), r["text"]] for r in results]

    # 8. Write output to /workspace/output/hybrid_rrf.json
    output_dir = "/workspace/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hybrid_rrf.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully wrote results to {output_path}")

if __name__ == "__main__":
    main()
