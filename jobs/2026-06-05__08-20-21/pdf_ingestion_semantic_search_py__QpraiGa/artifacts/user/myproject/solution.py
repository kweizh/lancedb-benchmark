import os
import lancedb
from openai import OpenAI
import httpx

# Configuration
DB_PATH = "/home/user/myproject/lancedb/"
ZEALT_RUN_ID = os.getenv("ZEALT_RUN_ID", "default")
TABLE_NAME = f"pdf_chunks_{ZEALT_RUN_ID}"

http_client = httpx.Client()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    http_client=http_client
)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def search(query: str, k: int) -> list[dict]:
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    
    query_embedding = get_embedding(query)
    
    results = table.search(query_embedding).limit(k).to_list()
    
    formatted_results = []
    for res in results:
        formatted_results.append({
            "doc_id": res["doc_id"],
            "page": int(res["page"]),
            "snippet": res["text"][:200]  # Short excerpt as required
        })
        
    return formatted_results
