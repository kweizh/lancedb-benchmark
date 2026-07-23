import os
import json
import psycopg2
import lancedb
import pyarrow as pa
import pyarrow.compute as pc

def _get_lancedb_tables():
    uri = os.environ["LANCEDB_URI"]
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    data_table_name = f"records_{run_id}"
    audit_table_name = f"audit_{run_id}"
    
    db = lancedb.connect(uri)
    data_table = db.open_table(data_table_name)
    audit_table = db.open_table(audit_table_name)
    return data_table, audit_table

def _write_critical_section(writer_id: str, rows: list, data_table, audit_table) -> int:
    # Determine the next sequence number from the current maximum sequence recorded in the audit table
    t = audit_table.search().select(['seq']).to_arrow()
    max_val = pc.max(t['seq']).as_py()
    if max_val is None:
        next_seq = 1
    else:
        next_seq = max_val + 1
        
    # Stamp each row with its writer_id and the assigned sequence number
    stamped_rows = []
    for r in rows:
        stamped_rows.append({
            'id': r['id'],
            'value': r['value'],
            'writer_id': writer_id,
            'seq': next_seq,
            'vector': r['vector']
        })
        
    # Upsert all rows into the LanceDB data table keyed by id
    new_data = pa.Table.from_pylist(stamped_rows, schema=data_table.schema)
    data_table.merge_insert("id") \
              .when_matched_update_all() \
              .when_not_matched_insert_all() \
              .execute(new_data)
              
    # Append exactly one record to the LanceDB audit table describing the operation
    value = rows[0]['value'] if len(rows) > 0 else 0
    ids_json = json.dumps([r['id'] for r in rows])
    audit_record = {
        'seq': next_seq,
        'writer_id': writer_id,
        'value': value,
        'ids_json': ids_json
    }
    audit_data = pa.Table.from_pylist([audit_record], schema=audit_table.schema)
    audit_table.add(audit_data)
    
    return next_seq

def guarded_write(writer_id: str, rows: list) -> int:
    lock_key = int(os.environ["PG_LOCK_KEY"])
    conn = psycopg2.connect()
    conn.autocommit = True
    cursor = conn.cursor()
    try:
        # Acquire PostgreSQL advisory lock, blocking
        cursor.execute("SELECT pg_advisory_lock(%s);", (lock_key,))
        
        # Inside critical section
        data_table, audit_table = _get_lancedb_tables()
        return _write_critical_section(writer_id, rows, data_table, audit_table)
    finally:
        try:
            cursor.execute("SELECT pg_advisory_unlock(%s);", (lock_key,))
        except Exception:
            pass
        finally:
            cursor.close()
            conn.close()

def try_write(writer_id: str, rows: list) -> int | None:
    lock_key = int(os.environ["PG_LOCK_KEY"])
    conn = psycopg2.connect()
    conn.autocommit = True
    cursor = conn.cursor()
    acquired = False
    try:
        # Acquire PostgreSQL advisory lock, non-blocking
        cursor.execute("SELECT pg_try_advisory_lock(%s);", (lock_key,))
        acquired = cursor.fetchone()[0]
        if not acquired:
            return None
            
        # Inside critical section
        data_table, audit_table = _get_lancedb_tables()
        return _write_critical_section(writer_id, rows, data_table, audit_table)
    finally:
        try:
            if acquired:
                cursor.execute("SELECT pg_advisory_unlock(%s);", (lock_key,))
        except Exception:
            pass
        finally:
            cursor.close()
            conn.close()
