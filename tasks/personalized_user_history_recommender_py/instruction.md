# Personalized User-History Recommender with LanceDB

## Background
A simple e-commerce recommender needs to personalize search results based on what each user has previously interacted with. The catalogue is stored in LanceDB and you must implement a Python script that blends a user's recent taste with the current query and runs a vector search against the catalogue, excluding items the user has already seen.

## Requirements
Implement a CLI script that:
- Loads up to the 10 most recent interactions for a user from the `user_history` LanceDB table.
- Joins the interacted item IDs back to the `items` table to retrieve their embeddings.
- Computes a "user taste vector" as the mean of those item embeddings.
- Blends the taste vector with the current query vector using a configurable blending coefficient `alpha`.
- Runs vector search on the `items` table against the blended vector.
- Excludes items the user has already interacted with from the returned candidates.
- Writes the top-k item IDs (in rank order) as a JSON array to an output file.

## Implementation Hints
- The LanceDB database lives at `/home/user/project/data`. Use `lancedb.connect(...)` and `open_table(...)` to access `items` and `user_history`.
- The `user_history` table has columns `user_id`, `item_id`, `rating`, `ts` (microsecond timestamp). Use the timestamp column to take the 10 most recent rows per user.
- The `items` table has columns `id`, `title`, `category`, and a 64-d `vector` column.
- The blending coefficient is read from the `BLEND_ALPHA` environment variable; default is `0.3`. The blended vector is `(1 - alpha) * query + alpha * taste`. When the user has no history, fall back to the pure query vector.
- Build the candidate query using LanceDB's vector search API and exclude already-seen items via a SQL `where` clause (for example a `NOT IN (...)` predicate over the user's seen item IDs).
- Read the query vector from a `.npy` file path passed on the command line and write results as a JSON list of integers.

## Acceptance Criteria
- Project path: /home/user/project
- Script path: /home/user/project/recommend.py
- Command: `python3 recommend.py --user-id <user_id> --query-vec <path_to_npy> --k <k> --output <path_to_json>`
- The script must read the blending coefficient from the `BLEND_ALPHA` environment variable (default `0.3`) and use it as documented above.
- The output file must contain a single JSON array of integer item IDs, length `k`, in rank order (best match first).
- None of the returned IDs may belong to the user's interaction history.
- The script must connect to the LanceDB database at `/home/user/project/data` and use the `items` and `user_history` tables that already exist there (do not recreate them).

