import unittest
import shutil
import tempfile
import os
import lancedb
from solution import TenantDB

class TestTenantDB(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.connection = lancedb.connect(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_and_list_tenants(self):
        # 1. Initially no tenants
        self.assertEqual(TenantDB.list_tenants(self.connection), [])

        # 2. Provision tenant_a
        db_a = TenantDB(self.connection, "tenant_a")
        rows_a = [
            {
                "id": "doc_a1",
                "text": "Tenant A document 1",
                "embedding": [0.1] * 32,
                "created_at": "2026-06-04T12:00:00Z"
            }
        ]
        tbl_a = db_a.create_documents_table(rows_a)
        self.assertIsNotNone(tbl_a)

        # List should now show tenant_a
        self.assertEqual(TenantDB.list_tenants(self.connection), ["tenant_a"])

        # 3. Provision tenant_b
        db_b = TenantDB(self.connection, "tenant_b")
        rows_b = [
            {
                "id": "doc_b1",
                "text": "Tenant B document 1",
                "embedding": [0.9] * 32,
                "created_at": "2026-06-04T12:01:00Z"
            }
        ]
        tbl_b = db_b.create_documents_table(rows_b)
        self.assertIsNotNone(tbl_b)

        # List should show both sorted
        self.assertEqual(TenantDB.list_tenants(self.connection), ["tenant_a", "tenant_b"])

    def test_cross_tenant_id_collision_on_create(self):
        db_a = TenantDB(self.connection, "tenant_a")
        rows_a = [
            {
                "id": "duplicate_id",
                "text": "Tenant A document",
                "embedding": [0.1] * 32,
                "created_at": "2026-06-04T12:00:00Z"
            }
        ]
        db_a.create_documents_table(rows_a)

        # Try to create tenant_b's table with the same id
        db_b = TenantDB(self.connection, "tenant_b")
        rows_b = [
            {
                "id": "duplicate_id",
                "text": "Tenant B document",
                "embedding": [0.2] * 32,
                "created_at": "2026-06-04T12:01:00Z"
            }
        ]
        with self.assertRaises(PermissionError):
            db_b.create_documents_table(rows_b)

        # Verify tenant_b's table was not created/registered
        self.assertEqual(TenantDB.list_tenants(self.connection), ["tenant_a"])

    def test_cross_tenant_id_collision_on_add(self):
        db_a = TenantDB(self.connection, "tenant_a")
        rows_a = [
            {
                "id": "doc_a1",
                "text": "Tenant A document 1",
                "embedding": [0.1] * 32,
                "created_at": "2026-06-04T12:00:00Z"
            }
        ]
        db_a.create_documents_table(rows_a)

        db_b = TenantDB(self.connection, "tenant_b")
        rows_b = [
            {
                "id": "doc_b1",
                "text": "Tenant B document 1",
                "embedding": [0.9] * 32,
                "created_at": "2026-06-04T12:01:00Z"
            }
        ]
        db_b.create_documents_table(rows_b)

        # Try to add to tenant_b with an id that belongs to tenant_a
        colliding_rows = [
            {
                "id": "doc_b2",
                "text": "Tenant B document 2",
                "embedding": [0.8] * 32,
                "created_at": "2026-06-04T12:02:00Z"
            },
            {
                "id": "doc_a1",  # Collides with tenant_a
                "text": "I am a spy",
                "embedding": [0.1] * 32,
                "created_at": "2026-06-04T12:03:00Z"
            }
        ]

        with self.assertRaises(PermissionError):
            db_b.add_documents(colliding_rows)

        # Verify that tenant_b's table was unchanged (i.e., doc_b2 was NOT added)
        results = db_b.search([0.9] * 32, 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_b1")

    def test_isolated_search(self):
        # Setup tenant_a
        db_a = TenantDB(self.connection, "tenant_a")
        rows_a = [
            {
                "id": "doc_a1",
                "text": "Tenant A close",
                "embedding": [0.1] * 32,
                "created_at": "2026-06-04T12:00:00Z"
            },
            {
                "id": "doc_a2",
                "text": "Tenant A far",
                "embedding": [0.5] * 32,
                "created_at": "2026-06-04T12:01:00Z"
            }
        ]
        db_a.create_documents_table(rows_a)

        # Setup tenant_b
        db_b = TenantDB(self.connection, "tenant_b")
        rows_b = [
            {
                "id": "doc_b1",
                "text": "Tenant B super close but isolated",
                "embedding": [0.01] * 32,
                "created_at": "2026-06-04T12:02:00Z"
            }
        ]
        db_b.create_documents_table(rows_b)

        # Search for tenant_a with query [0.0] * 32.
        # It should return tenant_a's docs in correct order, and NEVER return tenant_b's doc.
        query = [0.0] * 32
        results = db_a.search(query, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc_a1")
        self.assertEqual(results[1]["id"], "doc_a2")

        # Verify it has id and text fields
        for r in results:
            self.assertIn("id", r)
            self.assertIn("text", r)
            self.assertIn("embedding", r)
            self.assertIn("created_at", r)

    def test_delete_tenant(self):
        db_a = TenantDB(self.connection, "tenant_a")
        db_a.create_documents_table([
            {
                "id": "doc_a1",
                "text": "Tenant A",
                "embedding": [0.1] * 32,
                "created_at": "2026-06-04T12:00:00Z"
            }
        ])

        db_b = TenantDB(self.connection, "tenant_b")
        db_b.create_documents_table([
            {
                "id": "doc_b1",
                "text": "Tenant B",
                "embedding": [0.2] * 32,
                "created_at": "2026-06-04T12:01:00Z"
            }
        ])

        # Delete tenant_a
        db_a.delete_tenant()

        # List should reflect deletion immediately
        self.assertEqual(TenantDB.list_tenants(self.connection), ["tenant_b"])

        # tenant_b should still be queryable
        results = db_b.search([0.2] * 32, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_b1")

        # Deleting again should not crash (idempotency)
        db_a.delete_tenant()

if __name__ == "__main__":
    unittest.main()
