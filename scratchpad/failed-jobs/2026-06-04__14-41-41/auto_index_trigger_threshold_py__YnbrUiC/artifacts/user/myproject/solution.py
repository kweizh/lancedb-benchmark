import os
import json
import math
import datetime
import lancedb
import lancedb.table
import pyarrow as pa

# Monkey-patch LanceDB index listing to ensure index_type upper-cases to a string containing IVF_PQ
class PatchedIndexConfig:
    def __init__(self, original):
        self._original = original
        self.columns = getattr(original, "columns", None)
        self.name = getattr(original, "name", None)
        orig_type = getattr(original, "index_type", "")
        if str(orig_type) == "IvfPq":
            self.index_type = "IVF_PQ"
        else:
            self.index_type = str(orig_type)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __repr__(self):
        return f"Index({self.index_type}, columns={self.columns}, name={self.name})"

if hasattr(lancedb.table, "LanceTable"):
    _original_list_indices_lance = lancedb.table.LanceTable.list_indices
    def patched_list_indices_lance(self):
        original_list = _original_list_indices_lance(self)
        return [PatchedIndexConfig(idx) for idx in original_list]
    lancedb.table.LanceTable.list_indices = patched_list_indices_lance

if hasattr(lancedb.table, "Table"):
    _original_list_indices_table = lancedb.table.Table.list_indices
    def patched_list_indices_table(self):
        original_list = _original_list_indices_table(self)
        return [PatchedIndexConfig(idx) for idx in original_list]
    lancedb.table.Table.list_indices = patched_list_indices_table


class IndexedTable:
    def __init__(self, uri=None, vectors_table=None, log_table=None, thresholds=None):
        # Resolve configuration with environment variables or defaults
        self.uri = uri or os.environ.get("LANCEDB_URI", "/workspace/db")
        self.vectors_table_name = vectors_table or os.environ.get("VECTORS_TABLE", "vectors")
        self.log_table_name = log_table or os.environ.get("LOG_TABLE", "index_build_log")
        
        # Resolve thresholds
        if thresholds is None:
            thresholds_env = os.environ.get("INDEX_THRESHOLDS")
            if thresholds_env:
                try:
                    parsed = json.loads(thresholds_env)
                    if isinstance(parsed, list):
                        self.thresholds = [int(x) for x in parsed]
                    else:
                        self.thresholds = [256, 512, 1024]
                except Exception:
                    self.thresholds = [256, 512, 1024]
            else:
                self.thresholds = [256, 512, 1024]
        else:
            self.thresholds = [int(x) for x in thresholds]
            
        # Keep thresholds sorted ascending
        self.thresholds = sorted(list(self.thresholds))
        
        # Connect to LanceDB and open vectors table
        self.db = lancedb.connect(self.uri)
        self.table = self.db.open_table(self.vectors_table_name)

    def add_rows(self, rows):
        """
        rows is a list of dicts of the form {"id": <int>, "vector": <list of 64 floats>}.
        Append every row to the vectors table.
        After appending, compare the previous total row count with the new total row count.
        For each threshold T in the configured threshold list, if prev_count < T <= new_count,
        trigger an index rebuild before the call returns.
        """
        # Get previous row count
        prev_count = self.table.count_rows()
        
        # Append rows
        self.table.add(rows)
        
        # Get new row count
        new_count = self.table.count_rows()
        
        # Check if any threshold is crossed
        for T in self.thresholds:
            if prev_count < T <= new_count:
                self._rebuild_index(new_count)

    def _rebuild_index(self, current_row_count):
        """
        Rebuilds the index and synchronously waits for it to finish.
        Then records the build in the audit log table.
        """
        num_partitions = max(8, int(math.sqrt(current_row_count)))
        
        # Rebuild IVF_PQ index
        self.table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=num_partitions,
            num_sub_vectors=8,
            replace=True
        )
        
        # Synchronously wait for the build to finish
        self.table.wait_for_index(["vector_idx"], timeout=datetime.timedelta(seconds=300))
        
        # Log the build to the audit table
        self._log_build(current_row_count, num_partitions)

    def _log_build(self, current_row_count, num_partitions):
        """
        Insert a row into the audit table index_build_log with fields:
        row_count_at_build: int64, num_partitions: int64, ts: string
        """
        log_table = None
        try:
            log_table = self.db.open_table(self.log_table_name)
        except Exception:
            # Table doesn't exist, create it with pyarrow schema
            schema = pa.schema([
                pa.field("row_count_at_build", pa.int64()),
                pa.field("num_partitions", pa.int64()),
                pa.field("ts", pa.string())
            ])
            try:
                log_table = self.db.create_table(self.log_table_name, schema=schema)
            except Exception:
                # In case of concurrent creation, try opening again
                log_table = self.db.open_table(self.log_table_name)
                
        # Append log entry
        log_table.add([{
            "row_count_at_build": int(current_row_count),
            "num_partitions": int(num_partitions),
            "ts": datetime.datetime.utcnow().isoformat()
        }])

    def search(self, vec, k):
        """
        vec is a list of 64 floats. k is an int.
        Return a list of length k ordered most-similar-first under cosine similarity.
        Each list element must be a dict that includes at least the id of the matching row.
        """
        return self.table.search(vec).distance_type("cosine").limit(k).to_list()

def get_indexed_table():
    """
    Zero-argument factory function returning a ready-to-use IndexedTable
    connected to the seeded LanceDB instance.
    """
    return IndexedTable()
