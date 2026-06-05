# Rate-Limited Query Proxy over LanceDB

## Background

LanceDB-backed retrieval services often face bursty traffic from a small number of heavy users that can crowd out everyone else. A simple defense is a **per-user token-bucket rate limiter** that lives *in front* of the vector store but reuses LanceDB itself to persist bucket state, so that the proxy can be restarted without giving every user a fresh full bucket.

Your job is to implement that rate-limited query proxy in Python. The vector store and a seeded `documents_${ZEALT_RUN_ID}` table are already provisioned. You only need to write the proxy class and persist bucket state in a new LanceDB table.

## Requirements

- Implement a `RateLimitedSearchProxy` class in `/home/user/myproject/solution.py` with the signature:
  - `RateLimitedSearchProxy(table, capacity: int = 10, refill_per_sec: float = 5.0, per_user: bool = True, clock=None)`
  - `search(user_id: str, query_vec, k: int) -> dict` returning `{"results": list, "throttled": bool, "retry_after_ms": int}`.
- The proxy uses a **token-bucket** algorithm:
  - Each user starts with `capacity` tokens.
  - Tokens refill continuously at `refill_per_sec` tokens/second up to `capacity`.
  - One `search()` call costs one token. If a token is available, run the vector search against the wrapped LanceDB table and return `throttled=False`, `retry_after_ms=0`, and the top-`k` results as a list of dicts.
  - If no token is available, return `throttled=True`, `results=[]`, and `retry_after_ms` equal to the number of milliseconds (ceiling, never negative, capped at 60_000) until at least one token will be available.
- Bucket state must be persisted in a LanceDB table named `rate_buckets_${ZEALT_RUN_ID}` (read the suffix from the `ZEALT_RUN_ID` environment variable) with the schema `user_id: utf8, tokens: float64, last_refill_ts: int64` (last_refill_ts is nanoseconds since epoch from a monotonic clock seam — see hints).
- The proxy must survive restart: re-instantiating `RateLimitedSearchProxy` against the same database must recover each user's remaining tokens (NOT refill all users to full capacity).
- A `clock` keyword (optional) lets callers inject a no-arg callable returning the current time in **seconds (float)**; defaults to `time.monotonic`. Use this seam in *all* internal time reads so tests can run deterministically.
- The class must be thread-safe enough to support a parallel test from two distinct `user_id`s without dropping calls (each user has an independent bucket — `per_user=True` is the only mode you need to support).

## Implementation Hints

- The seeded `documents_${ZEALT_RUN_ID}` table is a 64-row LanceDB table with schema `{id: int64, text: utf8, vector: fixed_size_list<float32, 16>}`. The vectors were generated with `numpy.random.default_rng(2026)`. You do not need to know the contents — just pass them through.
- `lancedb.connect("/home/user/myproject/data")` opens the on-disk database the verifier expects you to use.
- Use `lancedb.connect(...).open_table("rate_buckets_<run_id>")` if the bucket table exists, otherwise create it with the schema above and an empty initial state.
- A reasonable persistence strategy is to flush on every state change with `merge_insert("user_id").when_matched_update_all().when_not_matched_insert_all().execute([row])`. The verifier's restart test will reopen the table after killing the proxy object.
- For the `retry_after_ms` calculation, the time until *one* full token = `(1.0 - current_tokens) / refill_per_sec` seconds; convert to ms with `math.ceil` and clamp to `[0, 60_000]`.
- Return shape for `results`: list of dicts with at least the fields `id` and `_distance`. You can call `tbl.search(query_vec).limit(k).to_list()` and pass the result through.
- Use a single `threading.Lock` to make the read-modify-write of bucket state safe under concurrent calls.

## Acceptance Criteria

- Project path: /home/user/myproject
- Solution file: /home/user/myproject/solution.py
- The proxy class is importable as `from solution import RateLimitedSearchProxy`.
- Bucket persistence table: `rate_buckets_${ZEALT_RUN_ID}` (suffix read from the `ZEALT_RUN_ID` environment variable) with schema `user_id: utf8, tokens: float64, last_refill_ts: int64`.
- A burst of 15 calls from a single user within < 100 ms of wall-clock time results in **exactly 10 successful calls** (`throttled=False`) and **exactly 5 throttled calls** (`throttled=True`).
- After waiting at least 0.6 s, the next 3 calls from the same user succeed.
- Two users called in parallel each get their own independent bucket (capacity is per-user, not shared).
- Persistence: After destroying the proxy instance and re-instantiating it without waiting, the recovered bucket reflects the prior consumption (not a full refill to capacity).

