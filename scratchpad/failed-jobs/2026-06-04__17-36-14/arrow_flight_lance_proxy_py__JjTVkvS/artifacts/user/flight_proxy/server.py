import json
import lancedb
import pyarrow.flight as flight

class LanceDBFlightServer(flight.FlightServerBase):
    def __init__(self, location, db_path, table_name, **kwargs):
        super().__init__(location, **kwargs)
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.tbl = self.db.open_table(self.table_name)

    def do_get(self, context, ticket):
        try:
            # Ticket body: a UTF-8 JSON object with the shape {"vector": [float, ... (length 32)], "k": int}
            ticket_bytes = ticket.ticket
            payload = json.loads(ticket_bytes.decode('utf-8'))
            
            vector = payload['vector']
            k = payload['k']
            
            # Query the real LanceDB table
            # We can re-open the table to ensure we always query the latest state of the table
            tbl = self.db.open_table(self.table_name)
            arrow_table = tbl.search(vector).limit(k).to_arrow()
            
            # Return the top-k results as an Arrow stream (RecordBatchStream)
            return flight.RecordBatchStream(arrow_table)
        except Exception as e:
            # You can raise a FlightError or standard Exception
            print(f"Error in do_get: {e}")
            raise flight.FlightServerError(f"Internal error: {str(e)}")

def main():
    location = "grpc://0.0.0.0:8815"
    db_path = "/home/user/flight_proxy/lancedb"
    table_name = "documents"
    
    print(f"Starting LanceDB Arrow Flight Server on {location}...")
    server = LanceDBFlightServer(location, db_path, table_name)
    print("Server is running. Press Ctrl+C to stop.")
    server.serve()

if __name__ == "__main__":
    main()
