import lancedb
import pyarrow as pa
import numpy as np
import os
import shutil
from solution import LoggedSearcher

def setup_test_db(db_uri, table_name):
    if os.path.exists(db_uri):
        shutil.rmtree(db_uri)
    
    db = lancedb.connect(db_uri)
    
    # Create articles table
    schema = pa.schema([
        ("id", pa.int64()),
        ("title", pa.string()),
        ("embedding", pa.list_(pa.float32(), 64))
    ])
    
    data = []
    for i in range(200):
        data.append({
            "id": i,
            "title": f"Article {i}",
            "embedding": np.random.randn(64).tolist()
        })
    
    db.create_table(table_name, data=data, schema=schema)
    return db

def test_logged_searcher():
    db_uri = "/tmp/test_lancedb"
    articles_table = "articles"
    logs_table = "query_logs"
    
    setup_test_db(db_uri, articles_table)
    
    searcher = LoggedSearcher(db_uri, articles_table, logs_table)
    
    # Test vector
    query_vec = np.random.randn(64).tolist()
    
    # First search (should create table)
    print("Performing first search...")
    results1 = searcher.search(query_vec, top_k=5, query_id="q1", user_id="u1", query_text="test query 1")
    assert len(results1) == 5
    
    # Second search (should append)
    print("Performing second search...")
    results2 = searcher.search(query_vec, top_k=3, query_id="q2", user_id="u2", query_text="test query 2")
    assert len(results2) == 3
    
    # Verify logs
    db = lancedb.connect(db_uri)
    log_table = db.open_table(logs_table)
    logs = log_table.to_arrow().to_pylist()
    
    assert len(logs) == 2
    
    # Check first log
    log1 = logs[0]
    assert log1['query_id'] == "q1"
    assert log1['user_id'] == "u1"
    assert log1['query_text'] == "test query 1"
    assert log1['hit_count'] == 5
    assert len(log1['top_ids']) == 5
    assert log1['latency_ms'] > 0
    assert [r['id'] for r in results1] == log1['top_ids']
    
    # Check second log
    log2 = logs[1]
    assert log2['query_id'] == "q2"
    assert log2['user_id'] == "u2"
    assert log2['query_text'] == "test query 2"
    assert log2['hit_count'] == 3
    assert len(log2['top_ids']) == 3
    assert log2['latency_ms'] > 0
    assert [r['id'] for r in results2] == log2['top_ids']
    
    # Check monotonicity of ts
    assert logs[1]['ts'] >= logs[0]['ts']
    
    print("All tests passed!")

if __name__ == "__main__":
    test_logged_searcher()
