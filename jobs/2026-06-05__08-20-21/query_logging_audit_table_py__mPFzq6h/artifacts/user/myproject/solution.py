import lancedb
import pyarrow as pa
import time
from datetime import datetime, timezone
from typing import List, Any

class LoggedSearcher:
    def __init__(self, db_uri: str, articles_table: str, logs_table: str):
        self.db_uri = db_uri
        self.articles_table_name = articles_table
        self.logs_table_name = logs_table
        self.db = lancedb.connect(self.db_uri)
        
    def _get_logs_schema(self):
        return pa.schema([
            ("query_id", pa.string()),
            ("user_id", pa.string()),
            ("query_text", pa.string()),
            ("ts", pa.timestamp("us", tz="UTC")),
            ("latency_ms", pa.float64()),
            ("hit_count", pa.int32()),
            ("top_ids", pa.list_(pa.int64()))
        ])

    def search(self, query_vector: List[float], top_k: int, query_id: str, user_id: str, query_text: str = "") -> List[dict]:
        # Start timer
        start_time = time.perf_counter()
        
        # Connect to articles table
        table = self.db.open_table(self.articles_table_name)
        
        # Perform search
        results = table.search(query_vector).limit(top_k).to_list()
        
        # End timer
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        # Ensure latency is strictly positive
        if latency_ms <= 0:
            latency_ms = 0.000001
            
        # Extract top_ids
        top_ids = [int(hit['id']) for hit in results]
        hit_count = len(top_ids)
        
        # Get current timestamp
        ts = datetime.now(timezone.utc)
        
        # Prepare log row
        log_row = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "ts": ts,
            "latency_ms": latency_ms,
            "hit_count": hit_count,
            "top_ids": top_ids
        }
        
        # Persist log row (lazy table creation)
        if self.logs_table_name not in self.db.table_names():
            self.db.create_table(self.logs_table_name, data=[log_row], schema=self._get_logs_schema())
        else:
            log_table = self.db.open_table(self.logs_table_name)
            log_table.add([log_row])
            
        return results
