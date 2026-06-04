import os
import sys
import json
import lancedb
import numpy as np

# Add project root to path
sys.path.append("/home/user/myproject")
import solution

def test_acceptance_criteria():
    # 1. Connect to original database to get ground truth
    db = lancedb.connect("/home/user/myproject/lancedb/")
    tbl_orig = db.open_table("articles")
    df_orig = tbl_orig.to_pandas()
    
    ids = df_orig["id"].values
    embeddings = np.stack(df_orig["embedding"].values).astype(np.float32)
    
    print(f"Loaded {len(embeddings)} original embeddings.")
    
    # Let's test with 10 different query vectors.
    # We can use some of the existing vectors in the dataset as query vectors to see how well they overlap.
    # We can also use some random vectors.
    np.random.seed(12345)
    query_indices = np.random.choice(len(embeddings), 10, replace=False)
    
    overlaps = []
    
    for idx in query_indices:
        query_vec = embeddings[idx]
        
        # Compute brute-force top-5 nearest neighbours in 128-d space using L2 distance
        # L2 distance: sum((x - y)^2)
        dists_128 = np.sum((embeddings - query_vec) ** 2, axis=1)
        # Get indices of top 5 smallest distances
        top5_indices_128 = np.argsort(dists_128)[:5]
        top5_ids_128 = set(ids[top5_indices_128])
        
        # Call search function from solution
        results_16 = solution.search(query_vec, 5)
        
        # Verify result structure
        assert len(results_16) == 5, f"Expected 5 results, got {len(results_16)}"
        for item in results_16:
            assert set(item.keys()) == {"id", "title", "original_id"}, f"Unexpected keys: {item.keys()}"
            assert isinstance(item["id"], int), "id is not int"
            assert isinstance(item["title"], str), "title is not str"
            assert isinstance(item["original_id"], int), "original_id is not int"
            
        # JSON serializability check
        try:
            json.dumps(results_16)
        except Exception as e:
            raise AssertionError(f"Results not JSON serializable: {e}")
            
        top5_ids_16 = set(item["original_id"] for item in results_16)
        
        overlap = len(top5_ids_128.intersection(top5_ids_16))
        overlaps.append(overlap)
        
        print(f"Query index {idx}:")
        print(f"  Brute-force 128-d top-5 IDs: {sorted(list(top5_ids_128))}")
        print(f"  PCA search 16-d top-5 IDs:   {sorted(list(top5_ids_16))}")
        print(f"  Overlap: {overlap}/5")
        
    print(f"Overlaps across 10 queries: {overlaps}")
    mean_overlap = np.mean(overlaps)
    min_overlap = np.min(overlaps)
    print(f"Mean overlap: {mean_overlap:.2f}, Min overlap: {min_overlap}")
    
    # Requirement: "the top-5 results in PCA space must overlap by at least 3 IDs with the brute-force top-5 nearest neighbours"
    # Wait, is this requirement per query or on average?
    # "for a precomputed query vector, the top-5 results in PCA space must overlap by at least 3 IDs with the brute-force top-5 nearest neighbours of that same query in the original 128-d space."
    # Let's make sure that for any typical query vector we have at least 3 ID overlap.
    assert min_overlap >= 3, f"Minimum overlap was {min_overlap}, which is less than 3!"
    print("All checks passed successfully!")

if __name__ == "__main__":
    test_acceptance_criteria()
