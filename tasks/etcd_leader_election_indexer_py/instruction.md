# Distributed LanceDB Indexer Coordinator with etcd Leader Election

## Background
A fleet of identical worker processes shares one LanceDB dataset. Table maintenance (compaction via `optimize()` plus vector-index rebuilds) is expensive and MUST be performed by exactly one worker at a time — otherwise workers race, waste resources, and can corrupt the picture of index health. You will build a coordinator that uses a **local etcd v3** instance for leader election so that only the elected leader runs maintenance while every follower stays idle. When a leader loses its lease (graceful step-down or crash), a new leader must take over without ever double-running maintenance.

etcd v3 is already installed and running locally in this environment. It exposes the gRPC protocol and its JSON gRPC-gateway on `http://127.0.0.1:2379` (e.g. `POST /v3/lease/grant`, `/v3/lease/keepalive`, `/v3/lease/revoke`, `/v3/kv/txn`, `/v3/kv/range`, `/v3/kv/put`, `/v3/kv/deleterange`). You may talk to etcd with any client you like; the pure-HTTP/JSON gateway with the `requests` library is the most dependency-light option (remember `key`/`value` fields are base64-encoded in JSON). Do NOT depend on any remote/cloud service — everything must run against the local etcd and the local LanceDB dataset.

## Requirements
- Implement lease-based leader election on top of etcd: a candidate grants a lease with a bounded TTL, then attempts an atomic compare-and-create transaction on a single election key bound to that lease. Only the winner becomes leader; the lease is kept alive automatically for as long as the leader is healthy.
- Provide a monotonically increasing **fencing token** for each leadership term so that a stale leader can be detected and rejected.
- Only the current leader may run LanceDB maintenance. Maintenance compacts fragments and rebuilds/updates the vector index on the shared table, must be idempotent (no rows lost), and must be fenced so a leader that has already lost leadership cannot run it.
- Support graceful failover (leader steps down and a follower takes over immediately) and ungraceful failover (leader crashes; the lease expires after its TTL and a follower then takes over). A new term's fencing token must always be strictly greater than every previous term's.

## Implementation Hints
- Use the etcd lease + a single election key with a create-revision (`CREATE`) transaction to guarantee single ownership; the election key's create-revision is a natural, globally-monotonic fencing token.
- Keep the lease alive from a background thread; on graceful resign revoke the lease and remove the key, on crash simply stop refreshing the lease and let it expire.
- Fence maintenance by re-checking, against etcd, that you still own the election key with your own term's token before touching the table.
- For the LanceDB side, run `table.optimize()` (it compacts small fragments and updates the vector index); read index health with `table.index_stats(<index_name>)` and versions with `table.list_versions()`.
- Project path: /home/user/myproject
- The coordinator must live in `/home/user/myproject/solution.py` and expose a class `IndexerCoordinator` with this constructor and defaults:
  `IndexerCoordinator(worker_id: str, *, election_key: str = "/lancedb/indexer/leader", ttl: int = 3, db_uri: str = "/home/user/myproject/lancedb", table_name: str = "documents", etcd_url: str = "http://127.0.0.1:2379")`.
- Required methods and their exact behavior:
  - `acquire() -> bool`: non-blocking attempt to become leader; returns `True` only if this instance won leadership, `False` otherwise. On failure it must not leak a lease.
  - `is_leader() -> bool`: `True` only while this instance currently holds leadership.
  - `fencing_token() -> int | None`: the integer token for this instance's current leadership term, or `None` if it does not currently hold leadership. Tokens are strictly increasing across successive terms.
  - `run_maintenance() -> dict`: if this instance is not the current leader, raise `PermissionError`; otherwise run maintenance on the shared table and return a dict with exactly the keys `worker_id`, `fencing_token`, `version_before`, `version_after`, `unindexed_before`, `unindexed_after` (worker_id is a str, the rest are ints).
  - `resign() -> None`: graceful step-down (revoke lease and delete the election key) so a follower can win immediately.
  - `simulate_crash() -> None`: stop refreshing the lease WITHOUT revoking it (models a crashed/partitioned leader); leadership must become available only after the TTL elapses.
  - `close() -> None`: release local resources/threads.
- The shared LanceDB table already exists at `db_uri`/`table_name` and is seeded with a trained vector index plus a batch of freshly-appended, not-yet-indexed rows so that maintenance has real work to do. Maintenance must operate on that exact table.

