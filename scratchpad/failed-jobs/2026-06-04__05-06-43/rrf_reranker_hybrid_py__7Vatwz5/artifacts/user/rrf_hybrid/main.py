import os
import json
import numpy as np
import pyarrow as pa
import lancedb
from lancedb.rerankers import RRFReranker

def main():
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    db = lancedb.connect(db_uri)
    
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
    
    vectors = np.random.default_rng(7).standard_normal((30, 8)).astype("float32")
    
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 8))
    ])
    
    data = []
    for i in range(30):
        data.append({
            "id": i,
            "text": texts[i],
            "vector": vectors[i].tolist()
        })
        
    table = db.create_table("kb", data=data, schema=schema, mode="overwrite")
    
    table.create_fts_index("text", use_tantivy=False, replace=True)
    
    query_vec = vectors[7]
    
    results = table.search(query_type="hybrid") \
        .vector(query_vec.tolist()) \
        .text("rrf reranker") \
        .rerank(RRFReranker()) \
        .limit(5) \
        .to_list()
        
    output = []
    for res in results:
        output.append([int(res["id"]), res["text"]])
        
    os.makedirs("/workspace/output", exist_ok=True)
    with open("/workspace/output/hybrid_rrf.json", "w") as f:
        json.dump(output, f)

if __name__ == "__main__":
    main()
