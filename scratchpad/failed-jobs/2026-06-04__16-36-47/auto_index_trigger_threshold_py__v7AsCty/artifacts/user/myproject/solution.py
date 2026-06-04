import os
import json
import math
import datetime
import lancedb

class IndexedTable:
    def __init__(self, db_uri, vectors_table_name, log_table_name, thresholds):
        self.db_uri = db_uri
        self.vectors_table_name = vectors_table_name
        self.log_table_name = log_table_name
        self.thresholds = sorted(thresholds)
        self.db = lancedb.connect(self.db_uri)
        self.table = self.db.open_table(self.vectors_table_name)
        
    def add_rows(self, rows):
        prev_count = self.table.count_rows()
        self.table.add(rows)
        new_count = self.table.count_rows()
        
        for t in self.thresholds:
            if prev_count < t <= new_count:
                num_partitions = max(8, int(math.sqrt(new_count)))
                self.table.create_index(
                    metric="cosine",
                    vector_column_name="vector",
                    index_type="IVF_PQ",
                    num_partitions=num_partitions,
                    num_sub_vectors=8,
                    replace=True
                )
                self.table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=300))
                
                log_row = {
                    "row_count_at_build": new_count,
                    "num_partitions": num_partitions,
                    "ts": datetime.datetime.utcnow().isoformat()
                }
                
                if self.log_table_name in self.db.table_names():
                    log_table = self.db.open_table(self.log_table_name)
                    log_table.add([log_row])
                else:
                    self.db.create_table(self.log_table_name, data=[log_row])

    def search(self, vec, k):
        return self.table.search(vec).distance_type("cosine").limit(k).to_list()

def get_indexed_table():
    thresholds = json.loads(os.environ.get("INDEX_THRESHOLDS", "[256, 512, 1024]"))
    db_uri = os.environ.get("LANCEDB_URI", "/workspace/db")
    vectors_table = os.environ.get("VECTORS_TABLE", "vectors")
    log_table = os.environ.get("LOG_TABLE", "index_build_log")
    
    return IndexedTable(
        db_uri=db_uri,
        vectors_table_name=vectors_table,
        log_table_name=log_table,
        thresholds=thresholds
    )
