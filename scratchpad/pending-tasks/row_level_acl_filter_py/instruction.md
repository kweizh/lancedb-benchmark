# Row-Level Access Control on Top of LanceDB Vector Search

## Background
You are wiring a retrieval layer for an enterprise knowledge base. The corpus already lives in a LanceDB table named `documents_${run-id}` where each document row has a 32-d float32 vector and three ACL columns: `owner_id` (string), `visibility` (string, one of `public` / `team` / `private`), and `team_id` (string). Each end-user is identified by a `user_id` string and carries a set of `roles` (e.g., `team:t_alpha`, `admin`). Your job is to make sure that when a user runs a similarity search, the results respect ACL semantics — and that the filtering happens **inside LanceDB**, not as a Python post-filter on the top-k candidates.

## Requirements
- Implement `ACLSearch` in `/home/user/myproject/solution.py` with the following public interface:
  - `ACLSearch(user_id: str, user_roles: Iterable[str], *, db_path: str | None = None, table_name: str | None = None)` — opens the seeded LanceDB table.
  - `search(query_vec: list[float] | numpy.ndarray, k: int) -> list[dict]` — returns up to `k` documents the caller is allowed to see, each entry having at least the keys `id` (int), `owner_id` (str), `visibility` (str), `team_id` (str), and the vector distance.
  - `build_where_clause() -> str` — returns the exact SQL `WHERE` clause string the instance uses for `tbl.search(...).where(..., prefilter=True)`. This is what the verifier inspects to confirm server-side filtering.
- Encode ACL semantics in a single SQL `where` clause:
  - `visibility = 'public'` is visible to everyone.
  - `visibility = 'team'` is visible only to users whose `roles` set contains an entry of the form `team:<team_id>` for that row's `team_id`.
  - `visibility = 'private'` is visible only when `owner_id == user_id`.
- The clause **MUST** be applied with `tbl.search(query_vec).where(clause, prefilter=True).limit(k)` — i.e., evaluated server-side before vector scoring.
- Read the LanceDB connection path from the `LANCE_DB_PATH` environment variable (default `/home/user/myproject/data`) and the table name from `documents_${ZEALT_RUN_ID}` where `ZEALT_RUN_ID` is provided in the environment.

## Implementation Hints
- Build the SQL clause as a disjunction of the three visibility rules. For the `team` rule, derive the set of team ids the user can access by parsing every role of the form `team:<team_id>` and emit a `team_id IN (...)` predicate (handle the empty-set case so the SQL stays valid).
- Strings inside the `where` clause must be quoted with single quotes; sanitize values so no caller can inject SQL (you may simply reject identifiers containing single quotes).
- LanceDB's SQL grammar supports `OR`, `AND`, `IN (...)`, and grouping via parentheses — see [`docs.lancedb.com/search/filtering`](https://docs.lancedb.com/search/filtering).
- Do not implement post-filtering in Python: filtering MUST happen in LanceDB. The verifier will inspect both the where-clause string and the source of `solution.py`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: python3 -c "from solution import ACLSearch" (importable as a module)
- The class `ACLSearch` exposes `search(query_vec, k)` and `build_where_clause()` as described above.
- `solution.py` source must contain a `prefilter=True` argument and a `.where(` call to confirm server-side filtering.
- For every test query, the set of `id` values returned by `search(...)` MUST equal the set of ids the verifier computes by brute force over the visible subset (numpy top-k by L2 distance).
- No row whose visibility/ownership rules forbid the calling user may ever be returned (zero leakage).
- Calling `search` with a user whose roles do not contain any `team:*` entry and whose `user_id` is not in `owner_id` MUST return only rows whose `visibility = 'public'`.

