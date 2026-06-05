import json
import pyarrow.flight as flight
import lancedb
import pyarrow as pa

class LanceDBFlightServer(flight.FlightServerBase):
    def __init__(self, location, db_path, table_name):
        super(LanceDBFlightServer, self).__init__(location)
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)

    def do_get(self, context, ticket):
        # Parse the JSON payload from ticket.ticket
        payload = json.loads(ticket.ticket.decode('utf-8'))
        vector = payload['vector']
        k = payload['k']

        # Run LanceDB vector search
        result_table = self.table.search(vector).limit(k).to_arrow()
        
        # Return the results as an Arrow stream
        return flight.RecordBatchStream(result_table)

if __name__ == '__main__':
    server = LanceDBFlightServer("grpc://0.0.0.0:8815", "/home/user/flight_proxy/lancedb", "documents")
    print("Starting Arrow Flight server on grpc://0.0.0.0:8815")
    server.serve()
