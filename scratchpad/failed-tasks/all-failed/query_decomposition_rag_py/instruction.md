# Query-Decomposition RAG over LanceDB

## Background
A LanceDB database at `/home/user/myproject/lancedb` has been pre-seeded with a 50-row `docs` table covering five distinct programming-language internals topics (10 documents per topic): Python GIL, Rust borrow checker, Go garbage collector, JavaScript event loop, and Java JIT compiler. Each row was embedded once at image build time with the real OpenAI `text-embedding-3-small` model. The table schema is:

- `id` (`int64`): a stable integer document id.
- `topic` (`string`): one of `python_gil`, `rust_borrow_checker`, `go_gc`, `javascript_event_loop`, `java_jit`.
- `content` (`string`): the document body (a short technical explanation).
- `embedding` (`fixed_size_list<float, 1536>`): the document's text-embedding-3-small vector.

A single-shot vector search against a compound, multi-aspect question tends to be dominated by whichever aspect has the strongest lexical/semantic signal, and therefore retrieves documents from only one or two topics. Your job is to build a **query-decomposition RAG** pipeline that first breaks the user's question into independent sub-questions with `gpt-4o-mini`, runs one retrieval per sub-question, and then fuses the results.

## Requirements
Implement a Python module `solution.py` at `/home/user/myproject/solution.py` exposing three callables:

1. `decompose(question: str) -> list[str]`
   - Calls the real OpenAI `gpt-4o-mini` chat completion API with `temperature=0`.
   - Instructs the model to output **exactly three distinct sub-questions**, one per line, that together cover the original question.
   - Returns a Python list of exactly three non-empty strings, in the order produced by the model.
2. `decomposed_search(question: str, k: int = 5) -> list[int]`
   - Calls `decompose(question)` to get the three sub-questions.
   - Embeds each sub-question individually with the real OpenAI `text-embedding-3-small` API.
   - For each sub-question vector, runs a top-`k` L2 vector search against the pre-seeded `docs` table.
   - Unions and de-duplicates the returned document ids, ranks them by their **minimum** distance across the three sub-question searches (smaller distance = better), and returns the top-`k` document ids as a `list[int]`.
3. `baseline_search(question: str, k: int = 5) -> list[int]`
   - Embeds the original `question` once with `text-embedding-3-small` and returns the top-`k` document ids from a single vector search against the same table.

## Implementation Hints
- Open the pre-seeded LanceDB database — do NOT recreate or re-seed the `docs` table.
- Use the official `openai` Python SDK and read `OPENAI_API_KEY` from the environment.
- For `decompose`, a strict system prompt that asks for exactly three sub-questions separated by newlines (no numbering, no extra commentary) makes parsing simple.
- LanceDB's `tbl.search(vec).limit(k).to_list()` returns rows with a `_distance` field you can use to rank the union across sub-question searches.
- Sub-question searches MAY share documents — make sure you union and de-duplicate before returning the final top-`k`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution module: /home/user/myproject/solution.py
- `decompose(question)` returns a list of exactly 3 non-empty distinct strings produced by a real `gpt-4o-mini` call.
- `decomposed_search(question, k)` returns a `list[int]` of length `k` (or fewer only if the union of sub-question top-`k` candidates contains fewer than `k` distinct documents), with no duplicate ids, ordered by ascending minimum distance across the three sub-question searches.
- `baseline_search(question, k)` returns a `list[int]` of length `k` with no duplicate ids, ordered by ascending distance from a single embedding of `question`.
- All embedding and chat-completion calls go to the real OpenAI API using `OPENAI_API_KEY`; no offline / mocked models.
- The candidate must NOT modify, drop, or rewrite the pre-seeded `docs` table.

