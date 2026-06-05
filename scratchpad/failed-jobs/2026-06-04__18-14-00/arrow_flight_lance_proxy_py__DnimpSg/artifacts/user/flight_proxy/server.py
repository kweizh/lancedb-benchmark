"""
Arrow Flight proxy server in front of a LanceDB vector table.

Exposes a single do_get endpoint:
  Ticket JSON: {"vector": [<32 floats>], "k": <int>}
  Response:    Arrow stream with columns id, text, embedding, _distance
               ordered by ascending _distance (nearest-neighbour first).

Bind: grpc://0.0.0.0:8815  (plain gRPC, no TLS)
"""

import json

import lancedb
import pyarrow as pa
import pyarrow.flight as flight

DB_PATH = "/home/user/flight_proxy/lancedb"
TABLE_NAME = "documents"
LISTEN_URI = "grpc://0.0.0.0:8815"

# Expected output column order and types.
RESULT_COLUMNS = ["id", "text", "embedding", "_distance"]


class LanceDBFlightServer(flight.FlightServerBase):
    """Arrow Flight server that proxies vector-search queries to LanceDB."""

    def __init__(self, location: str, db_path: str, table_name: str) -> None:
        super().__init__(location)
        self._db = lancedb.connect(db_path)
        self._table = self._db.open_table(table_name)
        print(f"[server] Opened '{table_name}' ({self._table.count_rows()} rows) at {db_path}")

    # ------------------------------------------------------------------
    # Flight RPC handlers
    # ------------------------------------------------------------------

    def do_get(self, context: flight.ServerCallContext, ticket: flight.Ticket):
        """
        Handle a top-k vector search request.

        Ticket payload (UTF-8 JSON):
            {"vector": [f0, f1, ..., f31], "k": <int>}

        Returns an Arrow RecordBatchStream with columns:
            id (string), text (string),
            embedding (fixed_size_list<float32>[32]), _distance (float32)
        """
        # --- 1. Parse the ticket ------------------------------------------
        try:
            payload = json.loads(ticket.ticket.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise flight.FlightServerError(f"Invalid ticket encoding: {exc}") from exc

        try:
            vector = payload["vector"]
            k = int(payload["k"])
        except (KeyError, TypeError, ValueError) as exc:
            raise flight.FlightServerError(
                f"Ticket must contain 'vector' (list of floats) and 'k' (int): {exc}"
            ) from exc

        if len(vector) != 32:
            raise flight.FlightServerError(
                f"'vector' must have exactly 32 elements, got {len(vector)}"
            )

        # --- 2. Run the LanceDB search ------------------------------------
        results: pa.Table = (
            self._table.search(vector)
            .limit(k)
            .to_arrow()
        )

        # --- 3. Ensure _distance is float32 and select required columns ---
        # LanceDB may return _distance as float64 on some versions; cast to f32.
        if "_distance" in results.schema.names:
            dist_idx = results.schema.get_field_index("_distance")
            if results.schema.field("_distance").type != pa.float32():
                results = results.set_column(
                    dist_idx,
                    "_distance",
                    results.column("_distance").cast(pa.float32()),
                )
        else:
            raise flight.FlightServerError("LanceDB result is missing '_distance' column")

        # Keep only the four required columns in a defined order.
        results = results.select(RESULT_COLUMNS)

        # --- 4. Stream back as RecordBatchStream --------------------------
        return flight.RecordBatchStream(results)


def main() -> None:
    server = LanceDBFlightServer(
        location=LISTEN_URI,
        db_path=DB_PATH,
        table_name=TABLE_NAME,
    )
    print(f"[server] Listening on {LISTEN_URI}")
    server.serve()


if __name__ == "__main__":
    main()
