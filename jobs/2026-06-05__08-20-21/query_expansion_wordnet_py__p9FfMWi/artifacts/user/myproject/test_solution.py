import os
import lancedb
from solution import expanded_search

def test_search():
    uri = os.environ.get("LANCEDB_URI")
    table_name = os.environ.get("LANCEDB_TABLE")
    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    
    # Try creating index first to ensure it's there
    try:
        table.create_fts_index("content", use_tantivy=False, replace=False)
    except:
        pass

    query = "car"
    
    # Plain search (baseline)
    # Note: We might need to handle the case where the baseline search fails if index isn't created yet,
    # but expanded_search handles index creation.
    baseline_results = table.search(query, query_type="fts").limit(10).to_list()
    baseline_ids = [int(res["id"]) for res in baseline_results]
    
    # Expanded search
    expanded_ids = expanded_search(query, k=10)
    
    print(f"Query: {query}")
    print(f"Baseline IDs: {baseline_ids}")
    print(f"Expanded IDs: {expanded_ids}")
    
    if set(baseline_ids) != set(expanded_ids):
        print("Success: Expanded search returned different results than baseline.")
    else:
        print("Warning: Expanded search returned same results as baseline for 'car'.")
        # Check if it found more results
        if len(expanded_ids) > len(baseline_ids):
             print(f"However, expanded found {len(expanded_ids)} while baseline found {len(baseline_ids)}")
        else:
             # Let's check some synonyms manually
             print("Checking 'automobile' directly in table...")
             auto_results = table.search("automobile", query_type="fts").limit(5).to_list()
             print(f"Automobile IDs: {[res['id'] for res in auto_results]}")

if __name__ == "__main__":
    test_search()
