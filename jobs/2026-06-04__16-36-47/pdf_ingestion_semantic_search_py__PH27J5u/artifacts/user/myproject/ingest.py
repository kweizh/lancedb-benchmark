import os
import glob
from pypdf import PdfReader
from openai import OpenAI
import lancedb
import pyarrow as pa

def get_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap
    return chunks

def main():
    client = OpenAI()
    
    # Read PDFs
    corpus_dir = "/app/corpus"
    pdf_files = glob.glob(os.path.join(corpus_dir, "*.pdf"))
    
    data = []
    chunk_id_counter = 0
    
    for pdf_file in pdf_files:
        doc_id = os.path.splitext(os.path.basename(pdf_file))[0]
        reader = PdfReader(pdf_file)
        
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text = page.extract_text()
            if not text:
                continue
            
            chunks = get_chunks(text)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                # Compute embedding
                response = client.embeddings.create(
                    input=chunk,
                    model="text-embedding-3-small"
                )
                embedding = response.data[0].embedding
                
                data.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_id": str(chunk_id_counter),
                    "text": chunk,
                    "embedding": embedding
                })
                chunk_id_counter += 1

    # Save to LanceDB
    run_id = os.environ.get("ZEALT_RUN_ID", "default")
    table_name = f"pdf_chunks_{run_id}"
    
    db_path = "/home/user/myproject/lancedb/"
    os.makedirs(db_path, exist_ok=True)
    
    db = lancedb.connect(db_path)
    
    if data:
        # Define schema explicitly or just let lancedb infer
        # Better to let it infer from list of dicts
        
        # Make ingestion idempotent
        if table_name in db.table_names():
            db.drop_table(table_name)
            
        db.create_table(table_name, data=data)
        print(f"Ingested {len(data)} chunks into {table_name}.")

if __name__ == "__main__":
    main()
