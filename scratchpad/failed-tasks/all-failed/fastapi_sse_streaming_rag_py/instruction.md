# FastAPI SSE Streaming RAG over LanceDB

## Background
You are building a small streaming Retrieval-Augmented-Generation (RAG) microservice. A LanceDB table at `/home/user/myproject/data/lancedb` has already been seeded with roughly 30 documentation chunks. Each row has fields `id` (string), `content` (string), and `vector` (1536-dim float vector produced from OpenAI `text-embedding-3-small` at build time). Your job is to expose a FastAPI service that retrieves relevant chunks for an incoming question and streams the language model's answer back to the client over Server-Sent Events (SSE).

## Requirements
- Implement a FastAPI application that exposes `POST /chat` and streams its response with `Content-Type: text/event-stream`.
- Embed the incoming `question` using **real** OpenAI `text-embedding-3-small` via the `openai` Python SDK (key in `OPENAI_API_KEY`).
- Retrieve the top-3 most similar rows from the pre-seeded LanceDB table.
- Generate the answer by calling **real** OpenAI `gpt-4o-mini` with `stream=True`, passing the retrieved chunks as context.
- Emit each streamed token as an SSE frame `event: token\ndata: <token text>\n\n`.
- Emit a final SSE frame `event: done\ndata: <json>\n\n` where `<json>` is a single line of JSON whose only top-level key is `sources` and whose value is the list of retrieved chunk `id` strings in retrieval order.

## Implementation Hints
- Open the table with `lancedb.connect("/home/user/myproject/data/lancedb").open_table("docs")`. The schema is fixed; do not re-create or overwrite it.
- Use `table.search(vector).limit(3).to_list()` (or `to_pandas`) to obtain the top-3 chunks and preserve retrieval order.
- For streaming OpenAI chat tokens, iterate over `client.chat.completions.create(model="gpt-4o-mini", stream=True, messages=[...])`. The text deltas are at `chunk.choices[0].delta.content`.
- A convenient way to build the SSE response is to write an `async` generator that yields properly formatted SSE frame strings and return it via `fastapi.responses.StreamingResponse(generator(), media_type="text/event-stream")`. `sse-starlette` is installed and may also be used.
- The streaming generator should yield at least one `event: token` frame even if the model returns a single delta; it is acceptable to split or coalesce model deltas as long as each emitted token frame contains real model text.
- The final `done` frame's `data:` payload must be valid JSON on a single line (no embedded newlines) so SSE parsers can read it in one frame.

## Acceptance Criteria
- Project path: /home/user/myproject
- Start command: uvicorn app:app --host 0.0.0.0 --port 8000
- Port: 8000
- API endpoint:
  - `POST /chat`

    ```json
    // Request body
    { "question": string }
    ```

    Response: `text/event-stream`. The stream MUST contain:
    - One or more frames of the form:
      ```
      event: token
      data: <token text>

      ```
    - Exactly one final frame:
      ```
      event: done
      data: {"sources": [<id_1>, <id_2>, <id_3>]}

      ```
      where `<id_i>` are the retrieved chunk IDs from the LanceDB `docs` table in retrieval order.
- Tokens MUST come from a real OpenAI `gpt-4o-mini` streaming call (no hard-coded text, no mocks).
- Retrieval MUST be done by embedding the user question with real OpenAI `text-embedding-3-small` and running a LanceDB vector search; the `sources` list MUST equal the IDs returned by that search in order.
- Do NOT regenerate or overwrite the seeded table at `/home/user/myproject/data/lancedb` (the verifier compares against the build-time fixture).

