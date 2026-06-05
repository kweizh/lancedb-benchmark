import sys
import os
import numpy as np
import lancedb

# Add current directory to path
sys.path.append("/home/user/myproject")

from solution import search

def test_search():
    os.environ["ZEALT_RUN_ID"] = "test_run"
    
    # Load original data to get a sample query
    db = lancedb.connect("/home/user/myproject/lancedb/")
    source_table = db.open_table("articles")
    sample_row = source_table.head(1).to_pandas().iloc[0]
    query_vec = sample_row['embedding']
    original_id = sample_row['id']
    
    print(f"Testing search with query from original_id: {original_id}")
    
    results = search(query_vec, 5)
    print(f"Search results: {results}")
    
    if not isinstance(results, list):
        print("Error: Results is not a list")
        return
    
    if len(results) == 0:
        print("Error: No results found")
        return

    # Check keys
    for res in results:
        for key in ["id", "title", "original_id"]:
            if key not in res:
                print(f"Error: Key {key} missing in result")
                return

    # The original vector should be among the top results (likely the first)
    top_original_ids = [res["original_id"] for res in results]
    if original_id in top_original_ids:
        print("Success: Original ID found in top results")
    else:
        print("Warning: Original ID not found in top results")

if __name__ == "__main__":
    test_search()
