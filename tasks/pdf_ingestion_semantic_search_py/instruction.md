# PDF Ingestion & Semantic Search with LanceDB

## Background
You are given a small corpus of three PDF documents bundled in this environment at `/app/corpus/`. Build a tiny semantic-search pipeline using **LanceDB** as the vector store and **OpenAI** embeddings. The pipeline must read the PDFs from the corpus directory, chunk and embed their text, store everything in a LanceDB table, and expose a Python `search(query, k)` function that returns the most semantically similar chunks.

The pipeline must work end-to-end against real services — no mocks of OpenAI or LanceDB, no local model inference.

## Requirements
- Read all PDF files from `/app/corpus/` (there are exactly three files).
- Extract text from each page of each PDF.
- Split the extracted text into chunks suitable for embedding (choose your own chunk size / overlap strategy).
- Compute embeddings for every chunk using the real OpenAI embedding API. The credentials are available in `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if set).
- Persist the chunks and their embeddings in a LanceDB table under `/home/user/myproject/lancedb/`. Each row must contain at least the fields `doc_id`, `page`, `chunk_id`, `text`, and `embedding`.
  - `doc_id` is the PDF file name without the `.pdf` extension (e.g. for `/app/corpus/alpha.pdf` use `doc_id = "alpha"`).
  - `page` is a 1-based integer page number identifying which page the chunk came from.
- Implement a function `search(query: str, k: int) -> list[dict]` in `/home/user/myproject/solution.py` that:
  - Embeds the `query` with the same OpenAI embedding model used during ingestion.
  - Returns the top-`k` most relevant chunks as a list of dicts, ordered by descending relevance.
  - Each result dict must contain the keys `doc_id` (str), `page` (int), and `snippet` (str — a short excerpt of the chunk text).

## Implementation Hints
- Use `pypdf` to extract page text. The `reportlab`-generated PDFs in `/app/corpus/` contain ordinary ASCII text.
- Use the official `openai` Python client to embed text. Pick an embedding model that is real and currently supported by the OpenAI Embeddings API.
- Use LanceDB's standard Python API (`lancedb.connect`, `create_table` / `open_table`, `table.search`) for storage and retrieval.
- Persist the table under `/home/user/myproject/lancedb/` so the verifier can re-open the same connection.
- To avoid collisions between concurrent runs, append the value of the `ZEALT_RUN_ID` environment variable to your table name (e.g. `pdf_chunks_${ZEALT_RUN_ID}`). The verifier reads the same env var and opens the same table.
- Make ingestion idempotent: when the script is re-run with the same `ZEALT_RUN_ID`, it should not error out and should leave the table queryable.

## Acceptance Criteria
- Project path: /home/user/myproject
- The candidate must provide `/home/user/myproject/solution.py` exposing a top-level callable `search(query: str, k: int) -> list[dict]`.
- Ingestion entrypoint: running `python3 /home/user/myproject/ingest.py` (or an equivalent script chosen by the candidate) must produce a LanceDB table at `/home/user/myproject/lancedb/` whose name is `pdf_chunks_${ZEALT_RUN_ID}` (where `${ZEALT_RUN_ID}` is read from the `ZEALT_RUN_ID` environment variable).
- After ingestion, `search(query, k)` must return a Python list of length `k` (assuming the table has at least `k` rows), each item a dict with the keys `doc_id` (str), `page` (int), and `snippet` (str).
- Result ordering: results must be ordered from most to least semantically relevant to the query.
- No mocks: both ingestion and `search()` must call the real OpenAI Embeddings API and the real LanceDB.

