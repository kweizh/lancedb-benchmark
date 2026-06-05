import os
import lancedb
from pypdf import PdfReader
from openai import OpenAI
import pandas as pd
import httpx

# Configuration
CORPUS_DIR = "/app/corpus/"
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

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def ingest():
    db = lancedb.connect(DB_PATH)
    
    data = []
    
    files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")]
    
    for filename in files:
        doc_id = filename.replace(".pdf", "")
        filepath = os.path.join(CORPUS_DIR, filename)
        
        print(f"Processing {filename}...")
        reader = PdfReader(filepath)
        
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            text = page.extract_text()
            if not text:
                continue
                
            chunks = chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                data.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_id": f"{doc_id}_{page_num}_{chunk_idx}",
                    "text": chunk,
                    "embedding": embedding
                })
    
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        # Assuming we want to overwrite or handle idempotency
        # The requirements say "when the script is re-run... it should not error out and should leave the table queryable."
        # Simplest way to ensure idempotency and cleanliness for this task is to drop and recreate or just append.
        # Given "make ingestion idempotent", I'll drop and recreate if it exists to ensure a fresh state for the current run.
        db.drop_table(TABLE_NAME)
    
    db.create_table(TABLE_NAME, data=data)
    print(f"Ingestion complete. Table '{TABLE_NAME}' created with {len(data)} chunks.")

if __name__ == "__main__":
    ingest()
