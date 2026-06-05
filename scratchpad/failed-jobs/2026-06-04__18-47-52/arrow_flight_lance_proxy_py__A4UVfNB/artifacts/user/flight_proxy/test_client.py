import json
import pyarrow.flight as flight
import pyarrow as pa
import numpy as np

def run_test():
    client = flight.connect("grpc://0.0.0.0:8815")
    
    vector = np.random.rand(32).astype(np.float32).tolist()
    k = 5
    
    ticket_payload = json.dumps({"vector": vector, "k": k}).encode('utf-8')
    ticket = flight.Ticket(ticket_payload)
    
    reader = client.do_get(ticket)
    table = reader.read_all()
    
    print(table.schema)
    print(f"Num rows: {table.num_rows}")
    print(table.to_pandas())

if __name__ == '__main__':
    run_test()
