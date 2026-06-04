import os
import lancedb
from openai import OpenAI

def search(query: str, k: int) -> list[dict]:
    """
    Embeds the query using OpenAI embeddings and searches the LanceDB table
    for the top-k most relevant chunks.
    
    Returns a list of dicts, where each dict has:
        - doc_id (str)
        - page (int)
        - snippet (str)
    """
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    db_dir = "/home/user/myproject/lancedb"
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"LanceDB directory not found at {db_dir}. Did you run ingestion first?")
        
    db = lancedb.connect(db_dir)
    table_name = f"pdf_chunks_{run_id}"
    
    try:
        tbl = db.open_table(table_name)
    except Exception as e:
        raise FileNotFoundError(f"Table {table_name} not found in LanceDB. Error: {e}")
        
    client = OpenAI()
    
    # Compute query embedding
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Perform vector search
    results = tbl.search(query_embedding).limit(k).to_list()
    
    # Format results
    formatted_results = []
    for r in results:
        formatted_results.append({
            "doc_id": r["doc_id"],
            "page": int(r["page"]),
            "snippet": r["text"]
        })
        
    return formatted_results
