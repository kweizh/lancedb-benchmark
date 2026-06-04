# Multi-Tenant Namespace Isolation with LanceDB

## Background
You are building the data layer for a SaaS product that stores per-customer document embeddings in a single shared LanceDB database. Each customer (tenant) must operate in a strictly isolated namespace: one tenant must never read another tenant's rows, and the same logical document id must not be claimed by two tenants. The goal is to implement a small wrapper class that enforces these guarantees on top of plain LanceDB tables.

## Requirements
Implement a `TenantDB` class in `/home/user/myproject/solution.py` that wraps a single LanceDB connection and enforces per-tenant isolation by physically separating each tenant's rows into a dedicated, prefixed table. The class must expose:

- A constructor `TenantDB(connection, tenant_id)` that binds the wrapper to one tenant.
- `create_documents_table(initial_rows)` to provision the tenant's documents table.
- `add_documents(rows)` that appends rows while guarding against cross-tenant id collisions.
- `search(query_vec, k)` that returns the top-`k` nearest documents for that tenant only.
- A class-level `list_tenants(connection)` lookup that enumerates all tenants currently present in the database.
- `delete_tenant()` that removes the tenant's namespace without touching others.

## Implementation Hints
- Use LanceDB Python (`lancedb.connect`) and store each tenant's rows in a table whose name is derived from the tenant id, so isolation is enforced at the table level rather than via filters.
- Document rows are dictionaries with `id`, `text`, `embedding` (a 32-dimensional float vector), and `created_at` (an ISO-8601 string). You decide the Arrow / LanceDB schema, but it must round-trip these fields losslessly.
- For the cross-tenant uniqueness check in `add_documents`, inspect the other tenants' tables in the same connection and raise `PermissionError` if any incoming `id` is already owned by another tenant. The check must not be defeatable by simply skipping it for tables that physically live elsewhere.
- For `list_tenants`, derive the result from `connection.table_names()` so newly created or deleted tenants are reflected immediately.
- For `search`, use LanceDB's native vector search on the tenant's own table; do not scan other tables.
- All vectors used in tests are precomputed; do not call any embedding model.

## Acceptance Criteria
- Project path: /home/user/myproject
- Solution file: /home/user/myproject/solution.py exporting a `TenantDB` class.
- Constructor signature: `TenantDB(connection, tenant_id)` where `connection` is a `lancedb.DBConnection` and `tenant_id` is a short lowercase string.
- Public methods on each instance:
  - `create_documents_table(initial_rows: list[dict]) -> Table` creates and returns the tenant's documents table populated with `initial_rows`.
  - `add_documents(rows: list[dict]) -> None` appends rows to the tenant's table and raises `PermissionError` when any row's `id` already exists in another tenant's namespace.
  - `search(query_vec, k: int) -> list[dict]` returns at most `k` rows, ordered by vector similarity, sourced exclusively from the calling tenant's table; each returned row must contain at least the `id` and `text` fields.
  - `delete_tenant() -> None` removes the tenant's namespace from the connection.
- Class-level helper: `TenantDB.list_tenants(connection) -> list[str]` returns the sorted list of tenant ids currently present in the connection, derived from `connection.table_names()`.
- Isolation guarantees that will be verified:
  - Each tenant's rows are stored in a physically separate table whose name encodes the tenant id, so an unrelated `connection.open_table(...)` call for another tenant never observes them.
  - `search(...)` for one tenant never returns rows belonging to another tenant.
  - `add_documents(...)` rejects cross-tenant id collisions with `PermissionError` and leaves the target table unchanged on rejection.
  - `delete_tenant()` removes only the calling tenant's namespace; other tenants remain queryable.

