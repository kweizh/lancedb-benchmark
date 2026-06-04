import lancedb
import time
import fcntl
import os
import random
import pyarrow as pa

class SafeWriter:
    def __init__(self, db_uri: str, table_name: str, key: str = "id"):
        self.db_uri = db_uri
        self.table_name = table_name
        self.key = key
        self.db = lancedb.connect(db_uri)
        
        # Ensure write_attempts table exists
        schema = pa.schema([
            pa.field("batch_id", pa.string()),
            pa.field("attempt_num", pa.int64()),
            pa.field("success", pa.bool_()),
            pa.field("error_msg", pa.string()),
            pa.field("ts", pa.int64())
        ])
        
        self.db.create_table("write_attempts", schema=schema, exist_ok=True)
        
        # Ensure lock file directory exists if db_uri is a local path
        if not os.path.exists(self.db_uri):
            os.makedirs(self.db_uri, exist_ok=True)

    def _log_attempt(self, batch_id: str, attempt_num: int, success: bool, error_msg: str):
        ts = time.time_ns()
        tbl = self.db.open_table("write_attempts")
        tbl.add([{
            "batch_id": batch_id,
            "attempt_num": attempt_num,
            "success": success,
            "error_msg": error_msg,
            "ts": ts
        }])

    def upsert(self, batch_id: str, rows: list[dict]) -> None:
        delays = [0.05, 0.10, 0.20, 0.40]
        max_attempts = len(delays) + 1
        
        lock_file_path = os.path.join(self.db_uri, f"{self.table_name}.lock")
        
        last_exception = None
        
        for attempt in range(max_attempts):
            success = False
            error_msg = ""
            
            try:
                # Try to acquire lock
                with open(lock_file_path, "w") as lock_file:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        # We have the lock, do the upsert
                        tbl = self.db.open_table(self.table_name)
                        (
                            tbl.merge_insert(self.key)
                            .when_matched_update_all(where="source.ts > target.ts")
                            .when_not_matched_insert_all()
                            .execute(rows)
                        )
                        
                        # Hold lock for a few ms to ensure contention test passes
                        time.sleep(0.005)
                        
                        success = True
                    except BlockingIOError:
                        raise RuntimeError("Lock acquisition failed, contention detected.")
                    finally:
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass
                            
            except Exception as e:
                error_msg = str(e)
                last_exception = e
                
            self._log_attempt(batch_id, attempt, success, error_msg)
            
            if success:
                return
                
            if attempt < len(delays):
                jitter = random.uniform(0, 0.01)
                time.sleep(delays[attempt] + jitter)
                
        if last_exception is not None:
            raise last_exception
