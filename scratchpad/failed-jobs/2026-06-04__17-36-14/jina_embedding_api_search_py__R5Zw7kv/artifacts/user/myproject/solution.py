import os
import json
import httpx
import pyarrow as pa
import lancedb

DB_DIR = "/home/user/myproject/lancedb_data"
HEADLINES_FILE = "/home/user/myproject/headlines.json"

def get_embeddings(texts: list[str], task: str) -> list[list[float]]:
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY environment variable is not set")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": task,
        "input": texts
    }
    
    response = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers=headers,
        json=payload,
        timeout=60.0
    )
    response.raise_for_status()
    resp_data = response.json()
    
    embeddings_data = resp_data["data"]
    if all("index" in x for x in embeddings_data):
        embeddings_data = sorted(embeddings_data, key=lambda x: x["index"])
    return [x["embedding"] for x in embeddings_data]

def build_index() -> None:
    # 1. Read headlines.json
    if not os.path.exists(HEADLINES_FILE):
        raise FileNotFoundError(f"Headlines file not found at {HEADLINES_FILE}")
        
    with open(HEADLINES_FILE, "r", encoding="utf-8") as f:
        headlines = json.load(f)
        
    # Extract headlines text to embed
    texts = [h["headline"] for h in headlines]
    
    # 2. Embed every headline using the Jina API with task="retrieval.passage"
    embeddings = get_embeddings(texts, task="retrieval.passage")
    
    # 3. Create / overwrite a LanceDB table named headlines_${ZEALT_RUN_ID} under DB_DIR
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    table_name = f"headlines_{run_id}"
    
    # Establish LanceDB connection
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)
    
    # Define schema using the returned embedding dimension
    dim = len(embeddings[0])
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("headline", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim))
    ])
    
    # Prepare data for LanceDB
    data = []
    for h, emb in zip(headlines, embeddings):
        data.append({
            "id": int(h["id"]),
            "headline": str(h["headline"]),
            "topic": str(h["topic"]),
            "vector": emb
        })
        
    table = pa.Table.from_pylist(data, schema=schema)
    db.create_table(table_name, data=table, mode="overwrite")

def search(query: str, k: int = 5, task: str = "retrieval.query") -> list[dict]:
    # 1. Embed query via Jina with the supplied task parameter
    query_embeddings = get_embeddings([query], task=task)
    query_embedding = query_embeddings[0]
    
    # 2. Connect to LanceDB and open the table
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    table_name = f"headlines_{run_id}"
    
    db = lancedb.connect(DB_DIR)
    tbl = db.open_table(table_name)
    
    # 3. Run vector search and return exactly k dicts
    results = tbl.search(query_embedding).limit(k).to_list()
    
    out = []
    for r in results:
        out.append({
            "id": int(r["id"]),
            "headline": str(r["headline"]),
            "topic": str(r["topic"])
        })
    return out
