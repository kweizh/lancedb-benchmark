import os
import lancedb
from openai import OpenAI

def search(query: str, k: int) -> list[dict]:
    client = OpenAI()
    
    # Compute query embedding
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"pdf_chunks_{run_id}"
    
    db_path = "/home/user/myproject/lancedb/"
    db = lancedb.connect(db_path)
    
    table = db.open_table(table_name)
    
    results = table.search(query_embedding).limit(k).to_list()
    
    output = []
    for r in results:
        # snippet is a short excerpt
        snippet = r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
        output.append({
            "doc_id": r["doc_id"],
            "page": r["page"],
            "snippet": snippet
        })
        
    return output
