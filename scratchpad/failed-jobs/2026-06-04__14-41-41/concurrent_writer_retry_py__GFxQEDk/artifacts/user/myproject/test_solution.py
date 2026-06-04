import os
import shutil
import threading
import random
import time
import lancedb
import pandas as pd
from solution import SafeWriter

def run_concurrent_test():
    src_db = "/home/user/lance_db"
    dest_db = "/tmp/test_lance_db"
    
    # Clean up and copy database
    if os.path.exists(dest_db):
        shutil.rmtree(dest_db)
    shutil.copytree(src_db, dest_db)
    
    # Initialize SafeWriter
    writer = SafeWriter(db_uri=dest_db, table_name="items")
    
    # We will run 10 concurrent threads
    # Each thread will write 10 batches of upserts
    # Each batch will target 5 random IDs between 0 and 199
    # Each batch will have a unique timestamp
    
    num_threads = 10
    batches_per_thread = 10
    
    # Track what we wrote so we can verify the largest ts for each ID
    # key: id, value: list of ts
    written_values = {}
    written_lock = threading.Lock()
    
    def worker(thread_id):
        for b in range(batches_per_thread):
            batch_id = f"thread_{thread_id}_batch_{b}"
            # Generate 5 random IDs
            ids = random.sample(range(200), 5)
            # Use current nanosecond timestamp
            ts = time.time_ns() + random.randint(1, 1000)
            
            rows = []
            for r_id in ids:
                rows.append({
                    "id": r_id,
                    "value": ts,
                    "ts": ts
                })
                
            # Upsert
            try:
                writer.upsert(batch_id, rows)
                # Log what we successfully wrote
                with written_lock:
                    for r_id in ids:
                        if r_id not in written_values:
                            written_values[r_id] = []
                        written_values[r_id].append(ts)
            except Exception as e:
                print(f"Upsert failed for {batch_id}: {e}")
                
            # Small sleep to mix things up
            time.sleep(random.uniform(0.05, 0.15))

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Now let's verify
    db = lancedb.connect(dest_db)
    tbl = db.open_table("items")
    df = tbl.to_pandas()
    
    # Check that there are exactly 200 rows and no duplicates
    assert len(df) == 200, f"Expected 200 rows, got {len(df)}"
    assert df["id"].nunique() == 200, "IDs are not unique or some are missing"
    
    # Check that the value for each ID is correct
    errors = 0
    for idx, row in df.iterrows():
        r_id = row["id"]
        val = row["value"]
        ts = row["ts"]
        
        # Get expected maximum ts for this ID
        expected_max_ts = 0
        if r_id in written_values:
            expected_max_ts = max(written_values[r_id])
            
        if expected_max_ts > 0:
            if val != expected_max_ts:
                print(f"Mismatch for ID {r_id}: expected value {expected_max_ts}, got {val}")
                errors += 1
            if ts != expected_max_ts:
                print(f"Mismatch for ID {r_id}: expected ts {expected_max_ts}, got {ts}")
                errors += 1
                
    # Check write_attempts table
    attempts_tbl = db.open_table("write_attempts")
    attempts_df = attempts_tbl.to_pandas()
    print("\n--- Write Attempts Summary ---")
    print(f"Total attempts recorded: {len(attempts_df)}")
    print(f"Successful attempts: {len(attempts_df[attempts_df['success'] == True])}")
    print(f"Failed attempts: {len(attempts_df[attempts_df['success'] == False])}")
    
    # Display some failed attempts to prove locking/retrying worked
    failures = attempts_df[attempts_df['success'] == False]
    if len(failures) > 0:
        print("\nFirst 5 failed attempts:")
        print(failures.head(5))
    else:
        print("\nNo failures recorded! (Contention might have been too low)")
        
    assert errors == 0, f"Found {errors} verification errors"
    print("\nAll checks passed successfully!")

def test_safe_writer():
    run_concurrent_test()

if __name__ == "__main__":
    run_concurrent_test()
