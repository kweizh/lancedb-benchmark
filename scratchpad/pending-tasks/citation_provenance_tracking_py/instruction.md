# Citation Provenance Tracking RAG

## Background
Build a citation-grounded RAG system on top of LanceDB. Given a natural-language query, the system retrieves text chunks from a curated knowledge base, asks `gpt-4o-mini` (via the `openai` SDK) to compose an answer that references chunks by id, then translates each cited chunk id back into the **exact character span** in the original source document so the caller receives verbatim quotes with provenance.

The environment ships with a fully-seeded LanceDB table called `chunks_${ZEALT_RUN_ID}` containing per-chunk character offsets (`span_start`, `span_end`, `doc_id`) and real OpenAI `text-embedding-3-small` (1536-d) vectors. The full source documents are baked into `/app/source_documents/` so the wrapper can re-read the raw text and extract the substring `[span_start:span_end]` for every cited chunk.

## Requirements
- Implement the function `answer(query: str, k: int) -> dict` in `/home/user/myproject/solution.py`.
- The returned dict must follow the shape:
  ```python
  {
    "answer": str,
    "citations": [
      {"doc_id": str, "span_start": int, "span_end": int, "quote": str},
      ...
    ]
  }
  ```
- Retrieve the top-`k` chunks from the seeded `chunks_${ZEALT_RUN_ID}` table using cosine vector search against the real OpenAI embedding of the query.
- Call `gpt-4o-mini` with a structured-output prompt that forces the LLM to cite each supporting chunk by its `chunk_id`. Parse the model's JSON output to obtain the answer text + chunk-id citations.
- For every chunk-id returned by the LLM, look up the corresponding row in the LanceDB table, copy `(doc_id, span_start, span_end)` from that row, read `/app/source_documents/<doc_id>.txt`, and set `quote` to the verbatim substring `text[span_start:span_end]`.
- If the model cannot answer from the retrieved context (off-topic query), return exactly `{"answer": "INSUFFICIENT_CONTEXT", "citations": []}`.

## Implementation Hints
- `ZEALT_RUN_ID` is provided in the environment; use it to build the table name `chunks_${ZEALT_RUN_ID}`.
- The LanceDB database lives under `/app/lancedb_data`.
- The seed already wrote 15 source documents to `/app/source_documents/<doc_id>.txt` and chunked each one into overlapping windows with `span_start`/`span_end` byte offsets that index directly into the on-disk text.
- Use `lancedb.connect("/app/lancedb_data")` then `db.open_table(...)`.
- Use the same model name `text-embedding-3-small` so that the query vector lives in the same 1536-d space as the seeded vectors.
- Structured outputs: ask `gpt-4o-mini` to return a JSON object with fields `{"answer": str, "chunk_ids": [str], "insufficient_context": bool}` (you may use `response_format={"type":"json_object"}`).
- Always resolve quotes by **reading the raw source file** — do **not** trust the LLM to copy substrings verbatim.
- When the LLM marks `insufficient_context: true` (or returns no chunk ids), short-circuit to the `INSUFFICIENT_CONTEXT` sentinel.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Module: `/home/user/myproject/solution.py` exposing `answer(query: str, k: int) -> dict`.
- The returned dict has top-level keys `answer` (str) and `citations` (list of dicts).
- For an in-corpus query, `len(citations) >= 2` and each citation has the exact keys `doc_id`, `span_start`, `span_end`, `quote`.
- For every citation, `quote == open(f"/app/source_documents/{doc_id}.txt").read()[span_start:span_end]` (byte-exact substring of the source).
- Every `(doc_id, span_start, span_end)` triple in `citations` corresponds to a row that was returned in the top-`k` LanceDB vector search results for the same query — no hallucinated chunk ids.
- An off-topic query (about a completely unrelated topic with no overlap with the seeded documents) returns exactly `{"answer": "INSUFFICIENT_CONTEXT", "citations": []}`.
- The seeded LanceDB table is named `chunks_${ZEALT_RUN_ID}` where `ZEALT_RUN_ID` is read from the environment.

