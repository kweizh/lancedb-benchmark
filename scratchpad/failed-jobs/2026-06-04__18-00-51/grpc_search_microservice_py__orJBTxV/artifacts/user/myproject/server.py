import sys
import os

# Ensure the project root is on the path so generated stubs resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import grpc
import lancedb

import search_pb2
import search_pb2_grpc

DB_PATH = "/home/user/myproject/data/lancedb"
TABLE_NAME = "documents"
LISTEN_ADDR = "0.0.0.0:50051"


class SearchServiceServicer(search_pb2_grpc.SearchServiceServicer):
    def __init__(self, table):
        self._table = table

    def Search(self, request, context):
        query_vector = list(request.vector)
        k = request.k
        where_clause = request.where_clause

        q = self._table.search(query_vector).limit(k)

        if where_clause:
            q = q.where(where_clause)

        results = q.to_list()

        hits = []
        for row in results:
            # LanceDB returns '_distance'; use it as score (non-negative float)
            distance = row.get("_distance", 0.0)
            score = float(distance) if distance is not None else 0.0
            hits.append(
                search_pb2.Hit(
                    id=int(row["id"]),
                    score=score,
                    title=str(row["title"]),
                )
            )

        return search_pb2.SearchResponse(hits=hits)


def serve():
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=4)
    )
    search_pb2_grpc.add_SearchServiceServicer_to_server(
        SearchServiceServicer(table), server
    )
    server.add_insecure_port(LISTEN_ADDR)
    server.start()
    print(f"Server listening on {LISTEN_ADDR}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
