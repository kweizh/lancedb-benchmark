import lancedb
import pyarrow as pa
from lancedb.table import Table

class TenantDB:
    def __init__(self, connection: lancedb.DBConnection, tenant_id: str):
        """
        Binds the wrapper to one tenant.
        
        :param connection: A lancedb.DBConnection object.
        :param tenant_id: A short lowercase string representing the tenant.
        """
        self.connection = connection
        self.tenant_id = tenant_id

    def _get_table_name(self) -> str:
        return f"tenant_{self.tenant_id}"

    def _check_cross_tenant_collisions(self, rows: list[dict]) -> None:
        if not rows:
            return
        
        incoming_ids = {row.get("id") for row in rows if row.get("id") is not None}
        if not incoming_ids:
            return

        all_tables = self.connection.table_names()
        current_table_name = self._get_table_name()
        other_tables = [name for name in all_tables if name.startswith("tenant_") and name != current_table_name]

        if not other_tables:
            return

        # Format SQL filter safely to prevent issues with quote characters
        escaped_ids = [str(val).replace("'", "''") for val in incoming_ids]
        if len(escaped_ids) == 1:
            filter_str = f"id = '{escaped_ids[0]}'"
        else:
            filter_str = f"id IN ({', '.join(f'\'{x}\'' for x in escaped_ids)})"

        for other_table_name in other_tables:
            other_tbl = self.connection.open_table(other_table_name)
            res = other_tbl.search().where(filter_str).to_arrow()
            if len(res) > 0:
                raise PermissionError("Document ID collision detected with another tenant.")

    def create_documents_table(self, initial_rows: list[dict]) -> Table:
        """
        Creates and returns the tenant's documents table populated with initial_rows.
        
        :param initial_rows: A list of document dictionaries.
        :return: The created lancedb.table.Table.
        """
        # First, check for cross-tenant collisions on initial_rows
        self._check_cross_tenant_collisions(initial_rows)

        table_name = self._get_table_name()
        schema = pa.schema([
            pa.field("id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), 32), nullable=False),
            pa.field("created_at", pa.string(), nullable=False),
        ])

        # Create table with the schema and initial rows
        return self.connection.create_table(table_name, data=initial_rows, schema=schema, mode="overwrite")

    def add_documents(self, rows: list[dict]) -> None:
        """
        Appends rows to the tenant's table and raises PermissionError when any row's id
        already exists in another tenant's namespace.
        
        :param rows: A list of document dictionaries to append.
        """
        # First, check for cross-tenant collisions on incoming rows
        self._check_cross_tenant_collisions(rows)

        table_name = self._get_table_name()
        tbl = self.connection.open_table(table_name)
        tbl.add(rows)

    def search(self, query_vec, k: int) -> list[dict]:
        """
        Returns at most k rows, ordered by vector similarity, sourced exclusively from the
        calling tenant's table; each returned row must contain at least the id and text fields.
        
        :param query_vec: A 32-dimensional float vector.
        :param k: Maximum number of rows to return.
        :return: A list of dictionaries.
        """
        table_name = self._get_table_name()
        tbl = self.connection.open_table(table_name)
        return tbl.search(query_vec).limit(k).to_list()

    def delete_tenant(self) -> None:
        """
        Removes the tenant's namespace from the connection without touching others.
        """
        table_name = self._get_table_name()
        if table_name in self.connection.table_names():
            self.connection.drop_table(table_name)

    @classmethod
    def list_tenants(cls, connection: lancedb.DBConnection) -> list[str]:
        """
        Returns the sorted list of tenant ids currently present in the connection.
        
        :param connection: A lancedb.DBConnection object.
        :return: A sorted list of tenant_id strings.
        """
        tenants = []
        for name in connection.table_names():
            if name.startswith("tenant_"):
                tenants.append(name[len("tenant_"):])
        return sorted(tenants)
