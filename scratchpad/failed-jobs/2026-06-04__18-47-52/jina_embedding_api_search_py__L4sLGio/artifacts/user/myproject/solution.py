import os
import json
import httpx
import pyarrow as pa
import lancedb

def get_embeddings(texts, task):
    api_key = os.environ.get("JINA_API_KEY")
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "jina-embeddings-v3",
        "task": task,
        "input": texts
    }
    response = httpx.post(url, headers=headers, json=data, timeout=60.0)
    response.raise_for_status()
    res_json = response.json()
    
    # Jina returns data in order, but let's be safe and sort by index
    sorted_data = sorted(res_json["data"], key=lambda x: x["index"])
    embeddings = [item["embedding"] for item in sorted_data]
    return embeddings

def build_index():
    with open("/home/user/myproject/headlines.json", "r") as f:
        headlines = json.load(f)
    
    texts = [item["headline"] for item in headlines]
    embeddings = get_embeddings(texts, task="retrieval.passage")
    
    data = []
    for hl, emb in zip(headlines, embeddings):
        data.append({
            "id": hl["id"],
            "headline": hl["headline"],
            "topic": hl["topic"],
            "vector": emb
        })
        
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("headline", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024))
    ])
    
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"headlines_{run_id}"
    
    db = lancedb.connect("/home/user/myproject/lancedb_data")
    db.create_table(table_name, data=data, schema=schema, mode="overwrite")

def search(query: str, k: int = 5, task: str = "retrieval.query") -> list[dict]:
    query_embedding = get_embeddings([query], task=task)[0]
    
    run_id = os.environ["ZEALT_RUN_ID"]
    table_name = f"headlines_{run_id}"
    
    db = lancedb.connect("/home/user/myproject/lancedb_data")
    tbl = db.open_table(table_name)
    
    results = tbl.search(query_embedding).limit(k).to_list()
    
    out = []
    for res in results:
        out.append({
            "id": res["id"],
            "headline": res["headline"],
            "topic": res["topic"]
        })
    return out
