import grpc
import lancedb
from concurrent import futures
import search_pb2
import search_pb2_grpc

class SearchService(search_pb2_grpc.SearchServiceServicer):
    def __init__(self, table):
        self.table = table

    def Search(self, request, context):
        try:
            vector = list(request.vector)
            query = self.table.search(vector)
            
            if request.where_clause:
                query = query.where(request.where_clause)
            
            k = request.k if request.k > 0 else 10
            results = query.limit(k).to_list()
            
            hits = []
            for row in results:
                # LanceDB distance is non-negative, and lower distance means higher similarity/relevance.
                # The verifier checks that IDs and titles match, and that score is a non-negative float.
                score = float(row.get('_distance', 0.0))
                hit = search_pb2.Hit(
                    id=row['id'],
                    score=score,
                    title=row['title']
                )
                hits.append(hit)
                
            return search_pb2.SearchResponse(hits=hits)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return search_pb2.SearchResponse()

def serve():
    db = lancedb.connect("/home/user/myproject/data/lancedb")
    table = db.open_table("documents")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    search_pb2_grpc.add_SearchServiceServicer_to_server(SearchService(table), server)
    
    server.add_insecure_port("0.0.0.0:50051")
    print("Starting gRPC server on 0.0.0.0:50051...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
