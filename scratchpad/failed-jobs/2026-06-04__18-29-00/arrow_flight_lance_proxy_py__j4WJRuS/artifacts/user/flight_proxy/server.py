"""Arrow Flight proxy in front of a LanceDB vector table."""

import json
import pyarrow as pa
import pyarrow.flight as flight
import lancedb


class LanceDBFlightServer(flight.FlightServerBase):
    """Expose the LanceDB ``documents`` table as an Arrow Flight do_get endpoint."""

    def __init__(self, location: str, db_path: str, **kwargs):
        super().__init__(location, **kwargs)
        self._db = lancedb.connect(db_path)
        self._table = self._db.open_table("documents")

    def do_get(self, context, ticket):
        """Handle a ``do_get`` request.

        The ticket bytes must be a UTF-8 JSON object with the shape:
            {"vector": [<32 floats>], "k": <int>}

        Returns an Arrow stream with columns id, text, embedding, _distance.
        """
        payload = json.loads(ticket.ticket.decode("utf-8"))
        query_vector = payload["vector"]
        k = payload["k"]

        result = self._table.search(query_vector).limit(k).to_arrow()
        return flight.RecordBatchStream(result)


if __name__ == "__main__":
    server = LanceDBFlightServer(
        location="grpc://0.0.0.0:8815",
        db_path="/home/user/flight_proxy/lancedb",
    )
    server.serve()