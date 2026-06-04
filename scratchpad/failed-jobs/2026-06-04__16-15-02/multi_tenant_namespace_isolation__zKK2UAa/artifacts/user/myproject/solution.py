"""Multi-tenant namespace isolation wrapper for LanceDB."""

from __future__ import annotations

import lancedb
import pyarrow as pa

# Prefix used to namespace tenant tables within the shared connection.
TENANT_PREFIX = "tenant_"

# Arrow schema for the documents table.  Defined explicitly so that
# round-tripping preserves types (especially created_at as a string
# rather than an automatic timestamp cast).
SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 32)),
        pa.field("created_at", pa.string()),
    ]
)


class TenantDB:
    """Wraps a LanceDB connection and enforces per-tenant isolation.

    Each tenant's rows live in a physically separate table whose name is
    ``f"tenant_{tenant_id}"``.  This guarantees that no tenant can ever
    observe another tenant's data, even through raw ``connection.open_table``
    calls for a different tenant id.
    """

    def __init__(self, connection: lancedb.DBConnection, tenant_id: str):
        self._connection = connection
        self._tenant_id = tenant_id
        self._table_name = f"{TENANT_PREFIX}{tenant_id}"

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_tenants(cls, connection: lancedb.DBConnection) -> list[str]:
        """Return the sorted list of tenant ids currently present in *connection*.

        Tenant ids are derived from the table name prefix, so newly created
        or deleted tenants are reflected immediately.
        """
        prefix_len = len(TENANT_PREFIX)
        # Use a large limit to ensure we get all tables even on cloud.
        names = list(connection.table_names(limit=10_000))
        tenant_ids = [
            name[prefix_len:]
            for name in names
            if name.startswith(TENANT_PREFIX)
        ]
        return sorted(tenant_ids)

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def create_documents_table(self, initial_rows: list[dict]):
        """Create and return the tenant's documents table populated with *initial_rows*."""
        # LanceDB create_table can accept data and will infer/create the schema.
        # We convert rows to match our explicit schema for consistency.
        records = self._normalise_rows(initial_rows)
        table = self._connection.create_table(self._table_name, records, schema=SCHEMA)
        return table

    def add_documents(self, rows: list[dict]) -> None:
        """Append *rows* to the tenant's table.

        Raises ``PermissionError`` if any row's ``id`` is already claimed by
        another tenant.  On rejection the target table is left unchanged.
        """
        incoming_ids = {str(row["id"]) for row in rows}

        # Cross-tenant uniqueness check: scan every other tenant table.
        prefix_len = len(TENANT_PREFIX)
        for table_name in self._connection.table_names(limit=10_000):
            if not table_name.startswith(TENANT_PREFIX):
                continue
            if table_name == self._table_name:
                continue

            other_tenant_id = table_name[prefix_len:]
            other_table = self._connection.open_table(table_name)

            # Collect all ids from the other tenant's table.
            other_ids = set(
                str(row["id"]) for row in other_table.to_arrow().to_pylist()
            )
            collision_ids = other_ids & incoming_ids
            if collision_ids:
                raise PermissionError(
                    f"Document id(s) {collision_ids} already belong to tenant "
                    f"'{other_tenant_id}'"
                )

        # No collisions — safe to append.
        records = self._normalise_rows(rows)
        table = self._connection.open_table(self._table_name)
        table.add(records)

    def search(self, query_vec, k: int) -> list[dict]:
        """Return at most *k* nearest documents for this tenant only.

        Each returned dict contains at least the ``id`` and ``text`` fields.
        """
        table = self._connection.open_table(self._table_name)
        results = table.search(query_vec).limit(k).to_list()
        return results

    def delete_tenant(self) -> None:
        """Remove only this tenant's namespace; other tenants remain queryable."""
        self._connection.drop_table(self._table_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_rows(rows: list[dict]) -> list[dict]:
        """Ensure every row conforms to the expected schema types.

        * ``id`` is coerced to ``str`` so that cross-tenant comparisons are
          reliable regardless of the original type.
        * ``embedding`` is coerced to a plain list of floats so LanceDB
          accepts it cleanly.
        * ``created_at`` is kept as-is (must already be an ISO-8601 string).
        """
        normalised = []
        for row in rows:
            normalised.append(
                {
                    "id": str(row["id"]),
                    "text": str(row["text"]),
                    "embedding": [float(v) for v in row["embedding"]],
                    "created_at": str(row["created_at"]),
                }
            )
        return normalised