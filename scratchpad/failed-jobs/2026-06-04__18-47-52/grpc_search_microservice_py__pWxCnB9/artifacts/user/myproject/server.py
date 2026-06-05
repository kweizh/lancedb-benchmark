import concurrent.futures
import grpc
import lancedb
import search_pb2
import search_pb2_grpc

class SearchServiceServicer(search_pb2_grpc.SearchServiceServicer):
    def __init__(self, table):
        self.table = table

    def Search(self, request, context):
        vector = list(request.vector)
        k = request.k
        where_clause = request.where_clause

        query = self.table.search(vector).limit(k)
        if where_clause:
            query = query.where(where_clause)

        results = query.to_list()
        
        hits = []
        for row in results:
            hits.append(search_pb2.Hit(
                id=row["id"],
                score=row.get("_distance", 0.0),
                title=row["title"]
            ))

        return search_pb2.SearchResponse(hits=hits)

def serve():
    db = lancedb.connect("/home/user/myproject/data/lancedb")
    table = db.open_table("documents")

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    search_pb2_grpc.add_SearchServiceServicer_to_server(SearchServiceServicer(table), server)
    server.add_insecure_port("0.0.0.0:50051")
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
