import lancedb

class TenantDB:
    def __init__(self, connection, tenant_id: str):
        self.connection = connection
        self.tenant_id = tenant_id
        self.table_name = f"tenant_{tenant_id}"

    @classmethod
    def list_tenants(cls, connection) -> list[str]:
        table_names = connection.table_names()
        tenants = []
        for name in table_names:
            if name.startswith("tenant_"):
                tenants.append(name[len("tenant_"):])
        return sorted(tenants)

    def _check_collisions(self, rows: list[dict]):
        incoming_ids = {row['id'] for row in rows}
        other_tenants = [t for t in self.list_tenants(self.connection) if t != self.tenant_id]
        for t in other_tenants:
            try:
                table = self.connection.open_table(f"tenant_{t}")
                existing_ids = set(table.to_arrow().column('id').to_pylist())
                if incoming_ids.intersection(existing_ids):
                    raise PermissionError("ID already owned by another tenant")
            except Exception as e:
                if isinstance(e, PermissionError):
                    raise e
                pass

    def create_documents_table(self, initial_rows: list[dict]):
        self._check_collisions(initial_rows)
        return self.connection.create_table(self.table_name, data=initial_rows)

    def add_documents(self, rows: list[dict]) -> None:
        self._check_collisions(rows)
        table = self.connection.open_table(self.table_name)
        table.add(rows)

    def search(self, query_vec, k: int) -> list[dict]:
        table = self.connection.open_table(self.table_name)
        return table.search(query_vec).limit(k).to_list()

    def delete_tenant(self) -> None:
        try:
            self.connection.drop_table(self.table_name)
        except Exception:
            pass
