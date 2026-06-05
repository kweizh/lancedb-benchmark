import concurrent.futures
import grpc
import lancedb
import search_pb2
import search_pb2_grpc


class SearchServiceServicer(search_pb2_grpc.SearchServiceServicer):
    def __init__(self, table):
        self.table = table

    def Search(self, request, context):
        query_vector = list(request.vector)
        k = request.k
        where_clause = request.where_clause

        query = self.table.search(query_vector).limit(k)

        if where_clause:
            query = query.where(where_clause)

        results = query.to_pandas()

        hits = []
        for _, row in results.iterrows():
            hit = search_pb2.Hit(
                id=int(row["id"]),
                score=float(row["_distance"]),
                title=str(row["title"]),
            )
            hits.append(hit)

        return search_pb2.SearchResponse(hits=hits)


def serve():
    db = lancedb.connect("/home/user/myproject/data/lancedb")
    table = db.open_table("documents")

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    search_pb2_grpc.add_SearchServiceServicer_to_server(
        SearchServiceServicer(table), server
    )
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("SearchService server started on 0.0.0.0:50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()