#!/usr/bin/env python3
import os
import sys
import json
import math
import hashlib
import pymysql
import pyarrow as pa
import lancedb
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent

CHECKPOINT_PATH = "/home/user/myproject/checkpoint.json"
LANCE_DB_DIR = "/home/user/myproject/lancedb"
LANCE_TABLE_NAME = "documents_index"

def get_vector(title, body):
    title = title or ""
    body = body or ""
    text = title + " " + body
    vector = [0.0] * 32
    tokens = text.lower().split()
    for token in tokens:
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 32
        vector[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector

def clean_row(row_values):
    return {
        "id": int(row_values["id"]),
        "title": str(row_values["title"]) if row_values.get("title") is not None else None,
        "body": str(row_values["body"]) if row_values.get("body") is not None else None,
        "category": str(row_values["category"]) if row_values.get("category") is not None else None,
        "price": float(row_values["price"]) if row_values.get("price") is not None else None,
        "vector": get_vector(row_values.get("title"), row_values.get("body"))
    }

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                data = json.load(f)
                return data.get("log_file"), data.get("log_pos")
        except Exception as e:
            print(f"Error loading checkpoint: {e}", file=sys.stderr)
    return None, None

def save_checkpoint(log_file, log_pos):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"log_file": log_file, "log_pos": log_pos}, f)

def get_earliest_binlog_info(mysql_settings):
    conn = pymysql.connect(
        host=mysql_settings["host"],
        port=mysql_settings["port"],
        user=mysql_settings["user"],
        password=mysql_settings["passwd"]
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW BINARY LOGS")
            rows = cursor.fetchall()
            if rows:
                return rows[0][0], 4
            else:
                raise Exception("No binary logs found")
    finally:
        conn.close()

def main():
    # Load MySQL settings from environment
    mysql_settings = {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", 3306)),
        "user": os.environ.get("MYSQL_USER", "cdc"),
        "passwd": os.environ.get("MYSQL_PASSWORD", "cdcpass")
    }

    # Initialize LanceDB
    db = lancedb.connect(LANCE_DB_DIR)
    schema = pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=True),
        pa.field("body", pa.string(), nullable=True),
        pa.field("category", pa.string(), nullable=True),
        pa.field("price", pa.float64(), nullable=True),
        pa.field("vector", pa.list_(pa.float32(), 32), nullable=False)
    ])
    table = db.create_table(LANCE_TABLE_NAME, schema=schema, exist_ok=True)

    # Load checkpoint or find earliest binlog
    log_file, log_pos = load_checkpoint()
    if not log_file:
        log_file, log_pos = get_earliest_binlog_info(mysql_settings)

    # Initialize BinLogStreamReader
    # Note: server_id must be different from MySQL server's own id (which is 1)
    stream = BinLogStreamReader(
        connection_settings=mysql_settings,
        server_id=100,
        only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
        only_tables=["documents"],
        only_schemas=["appdb"],
        resume_stream=True,
        log_file=log_file,
        log_pos=log_pos,
        blocking=False
    )

    inserts = 0
    updates = 0
    deletes = 0

    current_log_file = log_file
    current_log_pos = log_pos

    try:
        for event in stream:
            # Track current log file and position
            current_log_file = stream.log_file
            current_log_pos = stream.log_pos

            if isinstance(event, WriteRowsEvent):
                rows_to_upsert = []
                for row in event.rows:
                    cleaned = clean_row(row["values"])
                    rows_to_upsert.append(cleaned)
                    inserts += 1
                if rows_to_upsert:
                    pa_table = pa.Table.from_pylist(rows_to_upsert, schema=schema)
                    table.merge_insert("id") \
                         .when_matched_update_all() \
                         .when_not_matched_insert_all() \
                         .execute(pa_table)

            elif isinstance(event, UpdateRowsEvent):
                rows_to_upsert = []
                for row in event.rows:
                    cleaned = clean_row(row["after_values"])
                    rows_to_upsert.append(cleaned)
                    updates += 1
                if rows_to_upsert:
                    pa_table = pa.Table.from_pylist(rows_to_upsert, schema=schema)
                    table.merge_insert("id") \
                         .when_matched_update_all() \
                         .when_not_matched_insert_all() \
                         .execute(pa_table)

            elif isinstance(event, DeleteRowsEvent):
                delete_ids = []
                for row in event.rows:
                    delete_ids.append(row["values"]["id"])
                    deletes += 1
                if delete_ids:
                    id_filter = f"id IN ({','.join(map(str, delete_ids))})"
                    table.delete(id_filter)
    except Exception as e:
        print(f"Error during streaming: {e}", file=sys.stderr)
        raise e
    finally:
        stream.close()

    # Save checkpoint
    save_checkpoint(current_log_file, current_log_pos)

    # Print summary JSON to stdout
    summary = {
        "inserts": inserts,
        "updates": updates,
        "deletes": deletes,
        "log_file": current_log_file,
        "log_pos": current_log_pos
    }
    print(json.dumps(summary))

if __name__ == "__main__":
    main()
