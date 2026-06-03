LanceDB operates as a unified system supporting both vector search and full-text search (FTS), which can be combined to improve retrieval accuracy.

You need to configure a hybrid search pipeline by creating an FTS index on an existing table's `document_text` column, and then execute a hybrid search query using Reciprocal Rank Fusion (RRF).

**Constraints:**
- Must explicitly call `table.create_fts_index()` on the target text column before searching.
- Must set `query_type="hybrid"` in the search call.
- Must chain the `.rerank()` method using `lancedb.rerankers.RRFReranker()`.