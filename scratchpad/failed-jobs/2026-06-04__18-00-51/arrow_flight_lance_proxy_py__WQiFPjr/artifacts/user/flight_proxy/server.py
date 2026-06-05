import json

import lancedb
import pyarrow as pa
import pyarrow.flight as flight


DB_PATH = "/home/user/flight_proxy/lancedb"
TABLE_NAME = "documents"
BIND_URI = "grpc://0.0.0.0:8815"


class FlightProxyServer(flight.FlightServerBase):
    """Arrow Flight server that exposes LanceDB vector search over gRPC."""

    def __init__(self, location: str, db_path: str, table_name: str, **kwargs):
        super().__init__(location, **kwargs)
        self._db = lancedb.connect(db_path)
        self._table = self._db.open_table(table_name)

    def do_get(self, context, ticket: flight.Ticket):
        """Handle a top-k vector search request.

        Ticket payload (UTF-8 JSON):
            {"vector": [<32 floats>], "k": <int>}

        Returns an Arrow stream with columns:
            id (string), text (string),
            embedding (fixed_size_list<float, 32>), _distance (float32)
        """
        payload = json.loads(ticket.ticket.decode("utf-8"))
        vector = payload["vector"]
        k = int(payload["k"])

        result: pa.Table = (
            self._table
            .search(vector)
            .limit(k)
            .to_arrow()
        )

        # Ensure only the required columns are returned, in a predictable order.
        required = ["id", "text", "embedding", "_distance"]
        result = result.select(required)

        return flight.RecordBatchStream(result)


def main():
    server = FlightProxyServer(BIND_URI, DB_PATH, TABLE_NAME)
    print(f"Arrow Flight server listening on {BIND_URI}")
    server.serve()


if __name__ == "__main__":
    main()
