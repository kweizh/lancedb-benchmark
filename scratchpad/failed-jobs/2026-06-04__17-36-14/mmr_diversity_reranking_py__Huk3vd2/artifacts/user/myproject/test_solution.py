import os
import unittest
import numpy as np
import pyarrow as pa
import lancedb
from solution import build_dataset, mmr_search

class TestMMRSolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure ZEALT_RUN_ID is present
        cls.run_id = os.environ.get("ZEALT_RUN_ID")
        if not cls.run_id:
            raise ValueError("ZEALT_RUN_ID environment variable is not set!")
        cls.table_name = f"mmr_docs_{cls.run_id}"
        cls.db_dir = "/app/db"
        
        # Build dataset if it doesn't exist
        build_dataset()
        cls.db = lancedb.connect(cls.db_dir)
        cls.tbl = cls.db.open_table(cls.table_name)

    def test_table_metadata(self):
        # 1. Exactly 120 rows
        all_rows = self.tbl.to_arrow().to_pylist()
        self.assertEqual(len(all_rows), 120)
        
        # id values 0..119 inclusive
        ids = sorted([r["id"] for r in all_rows])
        self.assertEqual(ids, list(range(120)))
        
        # cluster_id values 0..9 inclusive, exactly 12 rows per cluster_id
        cluster_counts = {}
        for r in all_rows:
            cid = r["cluster_id"]
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            
        self.assertEqual(sorted(cluster_counts.keys()), list(range(10)))
        for cid in range(10):
            self.assertEqual(cluster_counts[cid], 12)
            
        # 2. Vector column type fixed_size_list<float32, 32>
        schema = self.tbl.schema
        vector_field = schema.field("vector")
        self.assertTrue(pa.types.is_fixed_size_list(vector_field.type))
        self.assertEqual(vector_field.type.list_size, 32)
        self.assertTrue(pa.types.is_float32(vector_field.type.value_type))

    def test_mmr_lambda_1_0(self):
        # 3. mmr_search(query_vec, k=10, lambda_=1.0) MUST equal top-10 pure cosine search
        rng = np.random.default_rng(2026)
        q = rng.standard_normal(32).tolist()
        
        # Pure top-10
        pure_top10 = [r["id"] for r in self.tbl.search(q).distance_type("cosine").limit(10).to_list()]
        
        # MMR with lambda=1.0
        mmr_res = mmr_search(q, k=10, lambda_=1.0)
        
        self.assertEqual(len(mmr_res), 10)
        self.assertEqual(mmr_res, pure_top10)

    def test_mmr_lambda_0_3(self):
        # 4. mmr_search(query_vec, k=10, lambda_=0.3) MUST return ids spanning >= 7 distinct cluster_id values
        # We need a query vector that has relevance to many clusters so that top-30 candidates span at least 7 clusters.
        # Let's generate Q using the same seed to get the centroids, and sum them to form a query vector.
        rng = np.random.default_rng(seed=2026)
        A = rng.standard_normal((32, 32))
        Q, _ = np.linalg.qr(A)
        q_combined = Q[:, :10].sum(axis=1).tolist()
        
        mmr_res = mmr_search(q_combined, k=10, lambda_=0.3)
        self.assertEqual(len(mmr_res), 10)
        self.assertEqual(len(set(mmr_res)), 10, "Should have no duplicate IDs")
        
        # Map id to cluster_id
        all_rows = self.tbl.to_arrow().to_pylist()
        id_to_cluster = {row["id"]: row["cluster_id"] for row in all_rows}
        
        clusters = [id_to_cluster[rid] for rid in mmr_res]
        distinct_clusters = set(clusters)
        print(f"lambda=0.3 selected clusters: {clusters} (count={len(distinct_clusters)})")
        self.assertGreaterEqual(len(distinct_clusters), 7)

    def test_mmr_lambda_0_7(self):
        # 5. mmr_search(query_vec, k=10, lambda_=0.7) MUST:
        # (a) return ids spanning >= 5 distinct cluster_id values
        # (b) include pure-vector-search top-1 id
        rng = np.random.default_rng(seed=2026)
        A = rng.standard_normal((32, 32))
        Q, _ = np.linalg.qr(A)
        q_combined = Q[:, :10].sum(axis=1).tolist()
        
        pure_top1 = self.tbl.search(q_combined).distance_type("cosine").limit(1).to_list()[0]["id"]
        
        mmr_res = mmr_search(q_combined, k=10, lambda_=0.7)
        self.assertEqual(len(mmr_res), 10)
        self.assertEqual(len(set(mmr_res)), 10, "Should have no duplicate IDs")
        
        # Map id to cluster_id
        all_rows = self.tbl.to_arrow().to_pylist()
        id_to_cluster = {row["id"]: row["cluster_id"] for row in all_rows}
        
        clusters = [id_to_cluster[rid] for rid in mmr_res]
        distinct_clusters = set(clusters)
        print(f"lambda=0.7 selected clusters: {clusters} (count={len(distinct_clusters)})")
        
        self.assertGreaterEqual(len(distinct_clusters), 5)
        self.assertIn(pure_top1, mmr_res)

    def test_general_requirements(self):
        # 6. Returned lists MUST have length 10, contain no duplicate ids, and every id MUST exist in the table.
        rng = np.random.default_rng(42)
        for _ in range(5):
            q = rng.standard_normal(32).tolist()
            for lam in [0.0, 0.2, 0.5, 0.8, 1.0]:
                res = mmr_search(q, k=10, lambda_=lam)
                self.assertEqual(len(res), 10)
                self.assertEqual(len(set(res)), 10)
                for rid in res:
                    self.assertTrue(0 <= rid < 120)

if __name__ == "__main__":
    unittest.main()
