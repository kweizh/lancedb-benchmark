import lancedb
import pyarrow as pa
import datetime
import time

class LoggedSearcher:
    def __init__(self, db_uri: str, articles_table: str, logs_table: str):
        self.db_uri = db_uri
        self.articles_table_name = articles_table
        self.logs_table_name = logs_table
        self.db = lancedb.connect(self.db_uri)
        # Open articles table
        self.articles_table = self.db.open_table(self.articles_table_name)
        self._logs_table = None

    def _get_or_create_logs_table(self):
        if self._logs_table is not None:
            return self._logs_table
        
        if self.logs_table_name in self.db.table_names():
            self._logs_table = self.db.open_table(self.logs_table_name)
        else:
            schema = pa.schema([
                ('query_id', pa.string()),
                ('user_id', pa.string()),
                ('query_text', pa.string()),
                ('ts', pa.timestamp('us')),
                ('latency_ms', pa.float64()),
                ('hit_count', pa.int64()),
                ('top_ids', pa.list_(pa.int64()))
            ])
            self._logs_table = self.db.create_table(self.logs_table_name, schema=schema)
        return self._logs_table

    def search(self, query_vector, top_k, query_id, user_id, query_text=""):
        # 1. Run vector similarity search and measure latency
        t_start = time.perf_counter()
        hits = self.articles_table.search(query_vector).limit(top_k).to_list()
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000.0
        if latency_ms <= 0:
            latency_ms = 0.000001  # strictly positive
            
        # 2. Extract returned ids
        top_ids = [int(hit['id']) for hit in hits]
        hit_count = len(top_ids)
        
        # 3. Create timestamp at logging time
        ts = datetime.datetime.now(datetime.timezone.utc)
        
        # 4. Get or create logs table
        logs_tbl = self._get_or_create_logs_table()
        
        # 5. Write exactly one audit row
        audit_row = {
            'query_id': str(query_id),
            'user_id': str(user_id),
            'query_text': str(query_text),
            'ts': ts,
            'latency_ms': float(latency_ms),
            'hit_count': int(hit_count),
            'top_ids': top_ids
        }
        logs_tbl.add([audit_row])
        
        return hits
