# Hybrid MinHash + Semantic Deduplication with LanceDB

## Background
Large corpora often contain near-duplicate documents that should be collapsed before training, indexing, or retrieval. Pure MinHash LSH (over text shingles) is fast but only sees lexical overlap, while pure semantic search via vector cosine is robust but quadratic. A common production pattern combines them: use MinHash LSH to cheaply propose candidate pairs, then **confirm** each candidate by reading the row from LanceDB and checking the precomputed embedding distance.

In this task you must build that pipeline against an already-seeded LanceDB table. The corpus contains 300 documents, each with a 64-dimensional precomputed cosine-clustered numpy vector. A subset of the documents are near-duplicates of each other (a small number of word-level substitutions) and the rest are unrelated singletons. The goal is to recover the correct connected components and expose them through a deterministic Python interface.

## Requirements
- Implement a Python module `solution.py` exposing a function `dedupe(db_uri: str, table_name: str) -> dict` that:
  1. Opens the LanceDB table at `db_uri` with the given `table_name`.
  2. Builds a `datasketch.MinHashLSH(threshold=0.7, num_perm=128)` over **3-shingles** (overlapping 3-word windows) of each row's `text` column.
  3. For every candidate pair surfaced by MinHash LSH, looks up both rows in LanceDB and **confirms** the pair only if the cosine distance between their stored vectors is `<= 0.10`. Use LanceDB's own cosine search/distance — do not recompute vectors in numpy if avoidable.
  4. Merges confirmed pairs into connected components (union-find or equivalent).
  5. Returns a dict with exactly two keys: `{"num_components": int, "components": list[list[int]]}` where every component is a sorted list of integer document `id`s and `components` itself is sorted by its first element. Singletons (documents with no confirmed duplicate) must appear as length-1 lists.
- Provide a CLI entry point `run.py` that:
  - Reads `ZEALT_RUN_ID` from the environment and constructs the table name as `documents_${ZEALT_RUN_ID}`.
  - Connects to the local LanceDB directory `/app/lancedb_data`.
  - Calls `solution.dedupe(...)` and writes the returned dict to `/home/user/myproject/result.json` (UTF-8, `indent=2`).
  - Prints a single line to stdout in the exact format `num_components=<int>`.

## Implementation Hints
- The build step seeded the table; do not try to regenerate or reorder the rows.
- A 3-shingle of the text `"the quick brown fox jumps"` is the multiset `{("the","quick","brown"), ("quick","brown","fox"), ("brown","fox","jumps")}`. Encode each shingle as UTF-8 bytes before calling `MinHash.update`.
- `MinHashLSH.query(m)` returns a list of previously-inserted keys whose Jaccard estimate exceeds the LSH threshold; insert all minhashes first, then query each one to surface candidate pairs.
- For the cosine confirmation, you can call `table.search(vec).distance_type("cosine").limit(K)` and filter by the `id` of the other row, or read the full embedding matrix once and compute pairwise cosine in numpy — either is acceptable as long as the threshold semantics (`<= 0.10`) match.
- Determinism: avoid Python set iteration order in your output. Sort within every component, and sort the outer `components` list by its first element.
- The MinHash library is `datasketch>=1.6,<2` (already installed in the environment).

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 run.py`
- The command must exit with code 0 and print exactly one line of the form `num_components=<int>` to stdout.
- After the command exits, the file `/home/user/myproject/result.json` must exist and contain a JSON object with the shape:

  ```json
  {
    "num_components": <int>,
    "components": [[<int>, ...], ...]
  }
  ```

- The total number of components must equal `250` for the seeded corpus (200 singletons + 50 collapsed duplicate pairs).
- Every confirmed duplicate pair must appear together exactly once in `components` (as a 2-element list of ids).
- The function `solution.dedupe(db_uri, table_name)` must be deterministic: two consecutive calls with the same arguments must return identical dicts (same key order/values for `num_components` and identical nested list structure for `components`).
- The candidate must use `datasketch.MinHashLSH(threshold=0.7, num_perm=128)` as the LSH backbone. Pure pairwise cosine without MinHash LSH will not satisfy the implementation contract checked by the test harness.
- Vectors and texts are read-only; the candidate must not modify, drop, or recreate the seeded table.

