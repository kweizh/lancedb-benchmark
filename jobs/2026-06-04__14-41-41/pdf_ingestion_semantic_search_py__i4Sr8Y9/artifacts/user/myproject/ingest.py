import os
import glob
import pypdf
import lancedb
from openai import OpenAI

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100) -> list[str]:
    # Clean up whitespace
    text = " ".join(text.split())
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks

def main():
    # Get run ID
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    table_name = f"pdf_chunks_{run_id}"
    print(f"Ingesting corpus into LanceDB table: {table_name}")
    
    # Read PDF files
    corpus_dir = "/app/corpus"
    pdf_files = sorted(glob.glob(os.path.join(corpus_dir, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {corpus_dir}")
    
    records = []
    for path in pdf_files:
        file_name = os.path.basename(path)
        doc_id = os.path.splitext(file_name)[0]
        print(f"Processing PDF: {path} (doc_id: {doc_id})")
        
        reader = pypdf.PdfReader(path)
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            text = page.extract_text() or ""
            chunks = chunk_text(text, chunk_size=400, overlap=100)
            
            for chunk_idx, chunk_text_str in enumerate(chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
                records.append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "text": chunk_text_str
                })
                
    if not records:
        print("No text chunks extracted from PDFs.")
        return
    
    print(f"Extracted {len(records)} chunks. Computing embeddings...")
    
    # Get OpenAI client
    client = OpenAI()
    
    # Compute embeddings in one batch
    texts_to_embed = [r["text"] for r in records]
    response = client.embeddings.create(
        input=texts_to_embed,
        model="text-embedding-3-small"
    )
    
    for idx, r in enumerate(records):
        r["embedding"] = response.data[idx].embedding
        
    # Connect to LanceDB
    db_dir = "/home/user/myproject/lancedb"
    os.makedirs(db_dir, exist_ok=True)
    db = lancedb.connect(db_dir)
    
    # Create table (idempotent overwrite)
    print(f"Writing to LanceDB table {table_name}...")
    db.create_table(table_name, data=records, mode="overwrite")
    print("Ingestion complete successfully.")

if __name__ == "__main__":
    main()
