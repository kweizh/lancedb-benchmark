import time
import datetime
import lancedb
import pyarrow as pa

class LoggedSearcher:
    def __init__(self, db_uri: str, articles_table: str, logs_table: str):
        self.db_uri = db_uri
        self.articles_table_name = articles_table
        self.logs_table_name = logs_table
        self.db = lancedb.connect(db_uri)

        self.log_schema = pa.schema([
            ("query_id", pa.string()),
            ("user_id", pa.string()),
            ("query_text", pa.string()),
            ("ts", pa.timestamp("us", tz="UTC")),
            ("latency_ms", pa.float64()),
            ("hit_count", pa.int64()),
            ("top_ids", pa.list_(pa.int64())),
        ])

    def search(self, query_vector, top_k, query_id, user_id, query_text=""):
        start_time = time.perf_counter()
        
        articles_table = self.db.open_table(self.articles_table_name)
        hits = articles_table.search(query_vector).limit(top_k).to_list()
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        if latency_ms <= 0:
            latency_ms = 1e-6
            
        top_ids = [hit["id"] for hit in hits]
        hit_count = len(top_ids)
        ts = datetime.datetime.now(datetime.timezone.utc)
        
        log_entry = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "ts": ts,
            "latency_ms": latency_ms,
            "hit_count": hit_count,
            "top_ids": top_ids
        }
        
        if self.logs_table_name not in self.db.table_names():
            self.db.create_table(self.logs_table_name, data=[log_entry], schema=self.log_schema)
        else:
            logs_table = self.db.open_table(self.logs_table_name)
            logs_table.add([log_entry])
            
        return hits
