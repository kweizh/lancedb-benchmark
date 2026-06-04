import os
import hashlib
import tempfile
import fcntl
import time
import random
import lancedb
import pyarrow as pa

class SafeWriter:
    def __init__(self, db_uri: str, table_name: str, key: str = "id"):
        self.db_uri = db_uri
        self.table_name = table_name
        self.key = key
        
        # Unique lock file path for the target table
        unique_str = hashlib.md5(f"{db_uri}_{table_name}".encode('utf-8')).hexdigest()
        self.lock_file_path = os.path.join(tempfile.gettempdir(), f"lancedb_lock_{unique_str}.lock")
        
        # Unique lock file path for the write_attempts table
        attempts_unique_str = hashlib.md5(f"{db_uri}_write_attempts".encode('utf-8')).hexdigest()
        self.attempts_lock_file_path = os.path.join(tempfile.gettempdir(), f"lancedb_lock_{attempts_unique_str}.lock")
        
        # Connect to LanceDB
        self.db = lancedb.connect(self.db_uri)
        
        # Ensure write_attempts table is created
        self._ensure_write_attempts_table()

    def _ensure_write_attempts_table(self):
        fd = os.open(self.attempts_lock_file_path, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            schema = pa.schema([
                pa.field('batch_id', pa.string()),
                pa.field('attempt_num', pa.int64()),
                pa.field('success', pa.bool_()),
                pa.field('error_msg', pa.string()),
                pa.field('ts', pa.int64())
            ])
            self.db.create_table('write_attempts', schema=schema, exist_ok=True)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _log_attempt(self, batch_id: str, attempt_num: int, success: bool, error_msg: str, ts: int):
        fd = os.open(self.attempts_lock_file_path, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            tbl = self.db.open_table("write_attempts")
            tbl.add([{
                "batch_id": batch_id,
                "attempt_num": attempt_num,
                "success": success,
                "error_msg": error_msg,
                "ts": ts
            }])
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def upsert(self, batch_id: str, rows: list[dict]) -> None:
        delays = [0.05, 0.10, 0.20, 0.40]
        last_exception = None
        
        for attempt in range(5):
            fd = os.open(self.lock_file_path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                # Try to acquire non-blocking lock
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Lock acquired, try to perform merge_insert
                try:
                    tbl = self.db.open_table(self.table_name)
                    tbl.merge_insert(self.key) \
                       .when_matched_update_all(where="source.ts > target.ts") \
                       .when_not_matched_insert_all() \
                       .execute(rows)
                    
                    # Hold lock for a short artificial duration (a few milliseconds)
                    time.sleep(0.002)
                    
                    # Log success
                    self._log_attempt(batch_id, attempt, success=True, error_msg="", ts=time.time_ns())
                    return
                except Exception as e:
                    self._log_attempt(batch_id, attempt, success=False, error_msg=str(e), ts=time.time_ns())
                    last_exception = e
            except BlockingIOError as e:
                self._log_attempt(batch_id, attempt, success=False, error_msg="Lock acquisition failed", ts=time.time_ns())
                last_exception = e
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception:
                    pass
                os.close(fd)
            
            # If we are here, the attempt failed. Sleep before retrying if retries are left
            if attempt < 4:
                base_delay = delays[attempt]
                # Small random jitter (using wide spread to avoid collision)
                sleep_time = random.uniform(0.5, 1.5) * base_delay
                time.sleep(sleep_time)
                
        # Raise last exception if all retries are exhausted
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("All retries exhausted without exception recorded")
