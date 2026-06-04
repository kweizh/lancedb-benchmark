# Streamlit LanceDB Explorer

## Background
You are building an internal browseable UI on top of a pre-existing LanceDB knowledge base. A LanceDB database has already been provisioned at `/app/db` and contains two tables of documents whose text content was pre-embedded with OpenAI's `text-embedding-3-small` model into a 1536-dimension `vector` column. Your job is to create a single-file Streamlit web application (`app.py`) that lets a non-engineer pick a table, run a free-text semantic search against it, see the top hits with their distance scores in a table, and drill into the full content of any individual row.

## Requirements
- Implement the explorer in a single Streamlit script at `/app/app.py`.
- Open the existing LanceDB database at `/app/db` (read-only is fine; do **not** overwrite or re-create the tables).
- Show a control that lets the user choose which table to search; the choices must come from the actual tables present in the database (no hard-coding).
- Show a free-text input where the user types a natural-language query.
- When a query is present, embed it with OpenAI's `text-embedding-3-small` model (read the API key from the `OPENAI_API_KEY` environment variable) and run a vector search against the currently selected table, returning the top 5 rows.
- Render the top 5 hits as a tabular view (Pandas `DataFrame`) that includes at minimum the `id`, `title`, and a `_distance` (or equivalent vector-distance) column ordered by ascending distance (best match first). Do not display the raw `vector` column.
- Provide a way for the user to inspect the full content of any individual hit; each top-K row's full text content (the `content` field) must be readable via an expandable region (e.g. `st.expander`).

## Implementation Hints
- Use the synchronous `lancedb` Python SDK (`lancedb.connect("/app/db")`, `db.table_names()`, `db.open_table(name)`).
- Use the official `openai` Python SDK (`from openai import OpenAI`) to call `client.embeddings.create(model="text-embedding-3-small", input=query)`.
- Streamlit components from the standard API are sufficient: `st.selectbox`, `st.text_input`, `st.dataframe`, `st.expander`.
- Each Streamlit run re-executes the whole script top to bottom, so be careful to only run the embedding + search when a non-empty query has been entered.
- Keep secrets out of the source: never hard-code an API key; always read it from the environment.
- The pre-seeded tables share a common schema: `id` (int64), `title` (string), `content` (string), `vector` (fixed-size list of 1536 float32 values).

## Acceptance Criteria
- Project path: /app
- Application entry point: /app/app.py
- Start command: streamlit run /app/app.py --server.headless true --server.port 8501
- Port: 8501
- Routes / UI features that will be checked:
  - The default page renders without raising a Python exception.
  - The page exposes a `selectbox` widget whose option list is exactly the set of tables present in the LanceDB database at `/app/db`.
  - The page exposes a `text_input` widget for the search query.
  - When the user picks a table and enters a non-empty query, the page renders a `st.dataframe` that contains at most 5 rows, including columns `id`, `title`, and a distance-score column (any column name containing `dist`, e.g. `_distance`), sorted in ascending order by that distance column.
  - For each of the top hits, the full `content` field of that row must be reachable from the rendered page (e.g. inside one or more `st.expander` blocks).
  - Before a query is entered, no search result rows are displayed.

