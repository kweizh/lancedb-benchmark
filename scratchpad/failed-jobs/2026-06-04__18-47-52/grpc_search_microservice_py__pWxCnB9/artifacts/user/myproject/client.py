import grpc
import search_pb2
import search_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = search_pb2_grpc.SearchServiceStub(channel)
        response = stub.Search(search_pb2.SearchRequest(
            vector=[0.1] * 24,
            k=5,
            where_clause="category = 'alpha'"
        ))
        print(response)

if __name__ == '__main__':
    run()
