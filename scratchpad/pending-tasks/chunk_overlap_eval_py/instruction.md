# Chunk Overlap Retrieval Quality Evaluation

## Background
You are building a retrieval-quality evaluation harness on top of **LanceDB** to measure how chunk overlap affects recall in a small RAG corpus. The corpus is three long source documents (text files in `/app/docs/`) and a fixed list of 10 anchor queries with ground-truth `(doc_id, char_span)` answers (`/app/anchors.json`). You will chunk each document at three different overlap values, embed the chunks with OpenAI, persist each chunk set into its own LanceDB table, and emit a per-overlap `Recall@5` and `MRR@5` report.

## Requirements
- For each overlap value in `{0, 75, 150}`, chunk every source document under `/app/docs/` with LangChain's `RecursiveCharacterTextSplitter` configured with `chunk_size=300` and `chunk_overlap=<value>`.
- Persist each chunk set into its own LanceDB table inside `/home/user/myproject/lancedb_data/`:
  - overlap 0 → table `chunks_o0_${ZEALT_RUN_ID}`
  - overlap 75 → table `chunks_o75_${ZEALT_RUN_ID}`
  - overlap 150 → table `chunks_o150_${ZEALT_RUN_ID}`
  Each row must carry, at minimum, the chunk text, the source document id, the chunk's start/end character offset inside the source document, and the embedding vector.
- Compute embeddings via OpenAI `text-embedding-3-small` (1536 dimensions). The `OPENAI_API_KEY` is exported in the environment at runtime — never bake it into the image.
- Read the 10 anchor queries and their ground-truth `(doc_id, span_start, span_end)` records from `/app/anchors.json`.
- For every overlap value, run a top-5 vector search per anchor query against the matching table and compute:
  - `Recall@5`: fraction of queries for which **at least one** of the 5 retrieved chunks is from the correct document **and** its `[start, end)` character interval overlaps the ground-truth interval (an intersection of length ≥ 1 counts).
  - `MRR@5`: the mean reciprocal rank of the first such hit per query (rank uses 1-indexed positions; if no hit in top 5, the reciprocal rank is 0).
- Write the final report to `/home/user/myproject/metrics.json` with this exact shape:
  ```json
  {
    "o0":   {"recall": <float>, "mrr": <float>},
    "o75":  {"recall": <float>, "mrr": <float>},
    "o150": {"recall": <float>, "mrr": <float>}
  }
  ```
  All four numeric values must be floats in `[0.0, 1.0]`.

## Implementation Hints
- Use `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=<value>, add_start_index=True)` to get character offsets for free; `add_start_index` adds `start_index` to the chunk metadata, which combined with `len(chunk_text)` gives you the chunk's `[start, end)` span.
- Connect with `lancedb.connect("/home/user/myproject/lancedb_data")`. Create one table per overlap value via `db.create_table(name, data=rows, mode="overwrite")`. The vector column can be a Python list of 1536 floats; LanceDB will infer a `fixed_size_list<float, 1536>`.
- Batch your embedding calls (e.g., 32 chunks per request) and reuse the same `openai.OpenAI()` client across batches. The total chunk count across all three tables is small (< 200), so a few sequential batches is plenty.
- For span overlap, two half-open intervals `[a, b)` and `[c, d)` intersect iff `a < d and c < b`. Enforce a same-document check before comparing intervals.
- Project path: `/home/user/myproject/`.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Command: `python3 /home/user/myproject/solution.py`
- After the command completes, the following must hold:
  - `/home/user/myproject/metrics.json` exists and matches the schema above (top-level keys `o0`, `o75`, `o150`; each maps to an object with float `recall` and `mrr` in `[0.0, 1.0]`).
  - The three LanceDB tables named `chunks_o0_${ZEALT_RUN_ID}`, `chunks_o75_${ZEALT_RUN_ID}`, `chunks_o150_${ZEALT_RUN_ID}` exist in `/home/user/myproject/lancedb_data/` and contain rows with at least the fields `text` (string), `doc_id` (string), `start` (int), `end` (int), `vector` (1536-d float list). The chunk-count for the `o150` table is strictly greater than the chunk-count for the `o0` table.
  - `metrics.json` reflects the documented invariant that overlap helps recall: `recall(o150) > recall(o0)`.
  - The verifier will re-embed one anchor query with `text-embedding-3-small`, run a top-5 search against one of the produced tables, and recompute Recall@5 for that table — the candidate's reported value must match within ±1e-6.
- The `ZEALT_RUN_ID` environment variable must be read at runtime and used as the table-name suffix.

