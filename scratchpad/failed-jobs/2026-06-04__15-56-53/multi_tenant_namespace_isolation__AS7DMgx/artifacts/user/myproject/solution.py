"""
solution.py – Multi-tenant namespace isolation wrapper for LanceDB.

Each tenant's documents are stored in a dedicated table whose name follows
the pattern:

    tenant_<tenant_id>__documents

This physical separation at the table level means:
  - No per-row filters are needed to achieve isolation.
  - Cross-tenant id uniqueness is enforced by scanning every *other* tenant
    table at write time.
  - list_tenants / delete_tenant are trivially derived from table names.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

# ---------------------------------------------------------------------------
# Arrow schema shared by all tenant tables
# ---------------------------------------------------------------------------
_DOCUMENTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8()),
        pa.field("text", pa.utf8()),
        pa.field("embedding", pa.list_(pa.float32(), 32)),
        pa.field("created_at", pa.utf8()),
    ]
)

# Naming conventions
_TABLE_PREFIX = "tenant_"
_TABLE_SUFFIX = "__documents"


def _table_name(tenant_id: str) -> str:
    """Return the LanceDB table name for a given tenant."""
    return f"{_TABLE_PREFIX}{tenant_id}{_TABLE_SUFFIX}"


def _tenant_id_from_table(table_name: str) -> str | None:
    """
    Extract the tenant id encoded in a table name, or return None if the
    table name does not follow the convention.
    """
    if table_name.startswith(_TABLE_PREFIX) and table_name.endswith(_TABLE_SUFFIX):
        inner = table_name[len(_TABLE_PREFIX) : -len(_TABLE_SUFFIX)]
        if inner:  # guard against an empty tenant id
            return inner
    return None


class TenantDB:
    """
    A per-tenant façade over a shared LanceDB connection.

    Parameters
    ----------
    connection:
        An open ``lancedb.DBConnection`` (obtained via ``lancedb.connect``).
    tenant_id:
        A short lowercase string that uniquely identifies the tenant.
        The value is embedded in every table name owned by this tenant, so it
        must not contain characters that are illegal in LanceDB table names.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, connection: Any, tenant_id: str) -> None:
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string.")
        self._conn = connection
        self._tenant_id = tenant_id
        self._table_name = _table_name(tenant_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_table(self) -> Any:
        """Open and return the tenant's documents table (must already exist)."""
        return self._conn.open_table(self._table_name)

    def _get_all_ids_in_other_tenants(self, incoming_ids: set[str]) -> dict[str, str]:
        """
        Scan every tenant table that belongs to a *different* tenant and
        return a mapping of ``{doc_id: owner_tenant_id}`` for any ``id``
        that appears in *incoming_ids*.

        This is the O(N_tenants) check that prevents cross-tenant id
        collisions.  The check iterates over table_names() at call time so
        it can never be bypassed simply by skipping a cached result.
        """
        collisions: dict[str, str] = {}

        for tname in self._conn.table_names():
            owner = _tenant_id_from_table(tname)
            if owner is None or owner == self._tenant_id:
                # Skip non-tenant tables and own table.
                continue

            other_table = self._conn.open_table(tname)
            # Pull only the id column – avoids transferring embeddings.
            try:
                rows = other_table.search().select(["id"]).to_list()
                existing_ids = {row["id"] for row in rows}
            except Exception:
                # If the table is empty or the column can't be read, skip.
                existing_ids = set()

            for doc_id in incoming_ids & existing_ids:
                collisions[doc_id] = owner

        return collisions

    # ------------------------------------------------------------------
    # Public instance API
    # ------------------------------------------------------------------

    def create_documents_table(self, initial_rows: list[dict]) -> Any:
        """
        Provision the tenant's documents table and populate it with
        *initial_rows*.

        Parameters
        ----------
        initial_rows:
            A list of row dicts, each containing ``id``, ``text``,
            ``embedding`` (32-element float list), and ``created_at``
            (ISO-8601 string).

        Returns
        -------
        The newly created ``lancedb.Table``.
        """
        if initial_rows:
            table = self._conn.create_table(
                self._table_name,
                data=initial_rows,
                schema=_DOCUMENTS_SCHEMA,
            )
        else:
            # Create an empty table using the schema directly.
            table = self._conn.create_table(
                self._table_name,
                schema=_DOCUMENTS_SCHEMA,
            )
        return table

    def add_documents(self, rows: list[dict]) -> None:
        """
        Append *rows* to the tenant's documents table.

        Before writing, every incoming ``id`` is checked against **all other
        tenants' tables** in the same connection.  If any collision is found,
        a ``PermissionError`` is raised and the tenant's table is left
        unchanged.

        Parameters
        ----------
        rows:
            A list of row dicts with the same shape as those passed to
            ``create_documents_table``.

        Raises
        ------
        PermissionError
            If one or more of the incoming ``id`` values already exist in
            another tenant's namespace.
        """
        if not rows:
            return

        incoming_ids = {row["id"] for row in rows}
        collisions = self._get_all_ids_in_other_tenants(incoming_ids)

        if collisions:
            details = ", ".join(
                f"'{doc_id}' owned by '{owner}'"
                for doc_id, owner in collisions.items()
            )
            raise PermissionError(
                f"Cross-tenant id collision detected – the following document "
                f"id(s) already belong to another tenant: {details}"
            )

        table = self._open_table()
        table.add(rows)

    def search(self, query_vec: list[float], k: int) -> list[dict]:
        """
        Return the top-*k* documents nearest to *query_vec* from this
        tenant's table only.

        Parameters
        ----------
        query_vec:
            A 32-dimensional float vector (list or array-like).
        k:
            Maximum number of results to return.

        Returns
        -------
        A list of dicts, each containing at least ``id`` and ``text`` (plus
        ``embedding``, ``created_at``, and ``_distance``).  Results are
        ordered by ascending distance.
        """
        table = self._open_table()
        results = (
            table.search(query_vec)
            .limit(k)
            .to_list()
        )
        return results

    def delete_tenant(self) -> None:
        """
        Drop the tenant's documents table from the connection.

        All other tenants' tables are left untouched.
        """
        self._conn.drop_table(self._table_name)

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_tenants(cls, connection: Any) -> list[str]:
        """
        Return a sorted list of all tenant ids currently present in
        *connection*, derived live from ``connection.table_names()``.

        Parameters
        ----------
        connection:
            An open ``lancedb.DBConnection``.

        Returns
        -------
        Sorted list of tenant id strings.
        """
        tenants: list[str] = []
        for tname in connection.table_names():
            tid = _tenant_id_from_table(tname)
            if tid is not None:
                tenants.append(tid)
        return sorted(tenants)
