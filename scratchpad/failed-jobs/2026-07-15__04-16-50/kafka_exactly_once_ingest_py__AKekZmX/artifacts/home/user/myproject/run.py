import os
import sys
import json
import time
import hashlib
import numpy as np
import pyarrow as pa
import lancedb
from confluent_kafka import Consumer, KafkaError, TopicPartition

class IngestionConsumer:
    def __init__(self, run_id: str, bootstrap_servers: str, db_path: str):
        self.run_id = run_id
        self.topic = f"ingest-docs-{run_id}"
        self.group_id = f"ingest-group-{run_id}"
        self.table_name = f"documents_{run_id}"
        self.db_path = db_path
        
        # Initialize LanceDB
        print(f"Connecting to LanceDB at: {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        self.schema = pa.schema([
            pa.field('id', pa.string(), nullable=False),
            pa.field('text', pa.string(), nullable=False),
            pa.field('source', pa.string(), nullable=False),
            pa.field('ts', pa.int64(), nullable=False),
            pa.field('partition', pa.int64(), nullable=False),
            pa.field('offset', pa.int64(), nullable=False),
            pa.field('vector', pa.list_(pa.float32(), 32), nullable=False)
        ])
        
        if self.table_name in self.db.table_names():
            print(f"Opening existing LanceDB table: {self.table_name}")
            self.table = self.db.open_table(self.table_name)
        else:
            print(f"Creating new LanceDB table: {self.table_name}")
            self.table = self.db.create_table(self.table_name, schema=self.schema)
            
        # Consumer state
        self.batch = []
        self.max_batch_size = 100
        self.idle_timeout_seconds = 5.0
        self.has_received_msg = False
        self.has_assigned_partitions = False
        self.last_msg_time = time.time()
        self.start_time = time.time()
        
        # Initialize Kafka Consumer
        print(f"Connecting to Kafka at: {bootstrap_servers}")
        conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        }
        self.consumer = Consumer(conf)
        
    def compute_embedding(self, text: str) -> list:
        vector = np.zeros(32, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            bucket = int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16) % 32
            vector[bucket] += 1.0
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
        
    def write_batch_and_commit(self):
        if not self.batch:
            return
            
        print(f"Processing batch of {len(self.batch)} messages...")
        rows_dict = {}
        latest_offsets = {}
        
        for msg in self.batch:
            if msg.error():
                continue
                
            val_bytes = msg.value()
            if val_bytes is None:
                continue
                
            try:
                val = json.loads(val_bytes.decode('utf-8'))
            except Exception as e:
                print(f"Failed to parse JSON: {e}")
                continue
                
            doc_id = val.get('id')
            text = val.get('text', '')
            source = val.get('source', '')
            ts = val.get('ts', 0)
            
            if not doc_id:
                key_bytes = msg.key()
                if key_bytes:
                    doc_id = key_bytes.decode('utf-8')
                else:
                    print("Message has no id and no key. Skipping.")
                    continue
                    
            doc_id = str(doc_id)
            partition = msg.partition()
            offset = msg.offset()
            
            vector = self.compute_embedding(text)
            
            # Keep only the latest message for each doc_id in this batch
            rows_dict[doc_id] = {
                'id': doc_id,
                'text': str(text),
                'source': str(source),
                'ts': int(ts),
                'partition': int(partition),
                'offset': int(offset),
                'vector': vector
            }
            latest_offsets[partition] = max(latest_offsets.get(partition, -1), offset)
            
        rows = list(rows_dict.values())
        if rows:
            new_data = pa.Table.from_pylist(rows, schema=self.schema)
            # Idempotent write using merge_insert
            print(f"Executing merge_insert for {len(rows)} unique documents...")
            (self.table.merge_insert(on="id")
                       .when_matched_update_all()
                       .when_not_matched_insert_all()
                       .execute(new_data))
            
            # Commit offsets manually AFTER writing to LanceDB
            offsets_to_commit = [TopicPartition(self.topic, p, o + 1) for p, o in latest_offsets.items()]
            try:
                self.consumer.commit(offsets=offsets_to_commit, asynchronous=False)
                print(f"Successfully processed and committed {len(rows)} unique documents (batch size: {len(self.batch)}).")
            except Exception as e:
                print(f"Warning: failed to commit offsets: {e}")
            
        self.batch = []
        
    def on_assign(self, consumer, partitions):
        print(f"Partitions assigned: {partitions}")
        self.has_assigned_partitions = True
        self.last_msg_time = time.time()
        
    def on_revoke(self, consumer, partitions):
        print(f"Partitions revoked: {partitions}")
        # Write and commit any pending batch before the partition assignment is revoked
        if self.batch:
            print("Writing pending batch before partition revocation...")
            try:
                self.write_batch_and_commit()
            except Exception as e:
                print(f"Error writing batch during revocation: {e}")
        self.has_assigned_partitions = False
        self.last_msg_time = time.time()
        
    def run(self):
        print(f"Subscribing to topic: {self.topic}")
        self.consumer.subscribe([self.topic], on_assign=self.on_assign, on_revoke=self.on_revoke)
        
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                # Update partition assignment status
                assignment = self.consumer.assignment()
                if len(assignment) > 0:
                    if not self.has_assigned_partitions:
                        print(f"Assignment detected: {assignment}. Resetting idle timer.")
                        self.has_assigned_partitions = True
                        self.last_msg_time = time.time()
                else:
                    if self.has_assigned_partitions:
                        print("No partitions currently assigned.")
                        self.has_assigned_partitions = False
                        self.last_msg_time = time.time()
                        
                if msg is None:
                    # No message received in poll timeout
                    if self.batch:
                        self.write_batch_and_commit()
                        
                    now = time.time()
                    if self.has_received_msg or self.has_assigned_partitions:
                        if now - self.last_msg_time >= self.idle_timeout_seconds:
                            print(f"No new messages received for {self.idle_timeout_seconds} seconds. Topic is drained.")
                            break
                    else:
                        # Waiting for partition assignment
                        if now - self.start_time >= 15.0:
                            print("Startup timeout of 15 seconds reached without any partition assignment. Exiting.")
                            break
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print(f"Reached end of partition {msg.partition()} at offset {msg.offset()}")
                    else:
                        print(f"Consumer error: {msg.error()}")
                    continue
                    
                # Valid message received
                self.has_received_msg = True
                self.last_msg_time = time.time()
                self.batch.append(msg)
                
                if len(self.batch) >= self.max_batch_size:
                    self.write_batch_and_commit()
                    
        except KeyboardInterrupt:
            print("Aborted by user.")
        finally:
            # Process any remaining messages in the batch
            if self.batch:
                try:
                    self.write_batch_and_commit()
                except Exception as e:
                    print(f"Error writing final batch during shutdown: {e}")
            self.consumer.close()
            print("Consumer closed cleanly.")

def main():
    # Read run ID
    run_id_path = '/logs/artifacts/run-id'
    if os.path.exists(run_id_path):
        with open(run_id_path, 'r') as f:
            run_id = f.read().strip()
    else:
        run_id = 'zrlocal'
        
    print(f"Using run_id: {run_id}")
    
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    db_path = '/home/user/myproject/lancedb'
    
    # Ensure parent directory of db_path exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    consumer = IngestionConsumer(run_id, bootstrap_servers, db_path)
    consumer.run()

if __name__ == '__main__':
    main()
