import os
import cohere

co = cohere.Client(os.environ["COHERE_API_KEY"])
response = co.embed(
    texts=["hello world"],
    model="embed-multilingual-v3.0",
    input_type="search_document"
)
print(type(response.embeddings))
if isinstance(response.embeddings, list):
    print("List of lists, len:", len(response.embeddings), type(response.embeddings[0]))
else:
    print(dir(response.embeddings))
