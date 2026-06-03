# Evaluation Dataset Research: LanceDB

## 1. Library Overview

*   **Description**: LanceDB is an open-source, AI-native multimodal vector database. It is built on top of the [Lance](https://github.com/lancedb/lance) data format, a modern columnar format designed for high-performance machine learning workloads and large-scale data storage. It supports vector search, full-text search (FTS), and SQL filtering in a single unified system.
*   **Ecosystem Role**: It functions as a "multimodal lakehouse," bridging the gap between traditional vector databases (like Pinecone) and data lakes. It is often used as the retrieval layer for RAG (Retrieval-Augmented Generation) applications, robotics datasets, and multimodal AI (images/video/audio).
*   **Project Setup**:
    *   **Python**: `pip install lancedb`
    *   **JavaScript/TypeScript**: `npm install @lancedb/lancedb`
    *   **Rust**: `cargo add lancedb`
    *   **Initialization**:
        ```python
        import lancedb
        db = lancedb.connect("data/my-db") # Local filesystem
        # For Enterprise/Cloud:
        # db = lancedb.connect("db://my-project", api_key="sk-...", region="us-east-1")
        ```

## 2. Core Primitives & APIs

### Connect & Table Management
*   **`connect(uri)`**: Establishes a connection to a local path or remote cloud instance.
*   **`create_table(name, data, schema, mode)`**: Creates a new table. Supports Pydantic models (Python) or Arrow schemas.
*   **`open_table(name)`**: Opens an existing table for querying.

### Search Primitives
*   **Vector Search**:
    ```python
    # Basic vector search
    results = table.search([0.1, 0.2, ...]).limit(10).to_pandas()
    ```
*   **Full-Text Search (FTS)**:
    ```python
    # Requires creating an FTS index first
    table.create_fts_index("text_column")
    results = table.search("search query").limit(10).to_list()
    ```
*   **Hybrid Search**:
    ```python
    from lancedb.rerankers import RRFReranker
    results = table.search("query text", query_type="hybrid") \
                   .rerank(RRFReranker()) \
                   .limit(10).to_pandas()
    ```

### Embedding API (Registry)
LanceDB manages embeddings automatically if a registry function is defined in the schema:
```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

func = get_registry().get("openai").create(name="text-embedding-3-small")

class MySchema(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()

table = db.create_table("my_table", schema=MySchema)
table.add([{"text": "hello world"}]) # Vector is generated automatically
```

**Documentation Links**:
*   [Quickstart](https://docs.lancedb.com/quickstart)
*   [Vector Search](https://docs.lancedb.com/search/vector-search)
*   [Full-Text Search](https://docs.lancedb.com/search/full-text-search)
*   [Hybrid Search](https://docs.lancedb.com/search/hybrid-search)
*   [Embedding API](https://docs.lancedb.com/embedding/)

## 3. Real-World Use Cases & Templates

*   **Multimodal RAG**: Storing raw image/video bytes in `pa.binary()` or `pa.large_binary()` columns along with embeddings. [Multimodal Guide](https://docs.lancedb.com/tables/multimodal).
*   **Time-Travel & Versioning**: Querying previous versions of a dataset for audit trails or reproducibility. [Versioning Guide](https://docs.lancedb.com/tables/versioning).
*   **Robotics/AV Datasets**: Managing massive frame-level data (e.g., KITTI, Waymo) using the Lance format's fast random access. [Robotics Tutorial](https://docs.lancedb.com/training/object-detection).
*   **Integration Templates**:
    *   [LlamaIndex Integration](https://docs.lancedb.com/integrations/ai/llamaIndex)
    *   [LangChain Integration](https://docs.lancedb.com/integrations/ai/langchain)

## 4. Developer Friction Points

*   **SQL Escaping in Filters**: Developers often encounter `Unterminated string literal` errors when using `table.update()` or `where()` clauses with strings containing single quotes (e.g., `I'm good`). [Issue #1429](https://github.com/lancedb/lancedb/issues/1429).
*   **Manual Optimization**: Unlike some cloud-native DBs, LanceDB OSS requires manual `table.optimize()` calls to merge small fragments and update indexes for newly added data. Forgetting this leads to slow performance over time.
*   **Strict Arrow Typing**: Since it is built on Arrow, schema mismatches (e.g., passing a float64 list to a float32 vector column) result in low-level errors that can be hard for beginners to debug.
*   **Local vs. Remote API Parity**: Some features (like specific `to_pandas` kwargs or the `Blob API`) behave differently or are not yet implemented in the Remote/Cloud SDK compared to the OSS library.

## 5. Evaluation Ideas

*   **Basic**: Implement a "Simple Document Search" that takes a directory of text files, embeds them using a local model, and performs a vector search.
*   **Intermediate**: Build a "Multimodal Product Gallery" where users can upload an image, store it as a blob in LanceDB, and find similar products using a CLIP-based multimodal embedding.
*   **Intermediate**: Create a "Version-Aware RAG" system that allows users to query documentation "as it was on [Date]" using LanceDB's versioning/snapshot features.
*   **Advanced**: Develop a "Hybrid Search Engine" that combines BM25 keyword matching and semantic vector search, including a custom reranker and handling complex metadata filters with special characters.
*   **Advanced**: Implement an "Auto-Maintenance Pipeline" for a high-churn dataset that monitors `num_unindexed_rows` and triggers `optimize()` and `compact_files()` asynchronously to maintain query latency.
*   **Advanced**: Configure a "Cloud-Stored Lakehouse" where LanceDB OSS is connected directly to an S3 bucket with custom storage options (credentials, region) and performs cross-region queries.

## 6. Sources

1.  [LanceDB Official Documentation](https://docs.lancedb.com/) - Primary source for all API and architectural details.
2.  [LanceDB llms.txt](https://docs.lancedb.com/llms.txt) - Structured index of the documentation.
3.  [LanceDB GitHub Issues](https://github.com/lancedb/lancedb/issues) - Source for developer friction points and bugs.
4.  [LanceDB Blog: Multimodal Lakehouse](https://lancedb.com/blog/multimodal-lakehouse/) - Context on ecosystem role.
5.  [Lance Format Documentation](https://lance.org/guide/blob/) - Details on the underlying storage engine and Blob API.