import json
import pyarrow.flight as flight

def main():
    try:
        # Connect to the flight server
        client = flight.FlightClient("grpc://localhost:8815")
        
        # Define a query vector (32 floats)
        # Let's just use some random or zero values for testing
        vector = [0.1] * 32
        k = 5
        
        # Construct the ticket
        payload = {
            "vector": vector,
            "k": k
        }
        ticket_bytes = json.dumps(payload).encode('utf-8')
        ticket = flight.Ticket(ticket_bytes)
        
        print(f"Sending ticket: {payload}")
        
        # Call do_get
        reader = client.do_get(ticket)
        
        # Read the resulting table
        table = reader.read_all()
        
        print("\nReceived table successfully!")
        print(f"Schema:\n{table.schema}")
        print(f"Row count: {table.num_rows}")
        print("\nData:")
        print(table.to_pandas() if hasattr(table, 'to_pandas') else table)
        
    except Exception as e:
        print(f"Error during client request: {e}")

if __name__ == "__main__":
    main()
