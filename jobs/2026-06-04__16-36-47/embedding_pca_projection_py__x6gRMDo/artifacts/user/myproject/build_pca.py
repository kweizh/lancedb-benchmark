import os
import lancedb
import numpy as np
import pyarrow as pa
from sklearn.decomposition import PCA

def main():
    db_path = "/home/user/myproject/lancedb"
    db = lancedb.connect(db_path)
    
    # Read articles
    table = db.open_table("articles")
    df = table.to_pandas()
    
    embeddings = np.stack(df['embedding'].values)
    
    # Fit PCA
    pca = PCA(n_components=16, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings).astype(np.float32)
    
    # Save PCA model
    os.makedirs("/app", exist_ok=True)
    np.savez("/app/pca_model.npz", components=pca.components_, mean=pca.mean_)
    
    # Create new table
    run_id = os.environ.get("ZEALT_RUN_ID", "test")
    table_name = f"articles_pca_{run_id}"
    
    df['embedding'] = list(embeddings_pca)
    df['original_id'] = df['id']
    
    # Ensure correct schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 16)),
        pa.field("original_id", pa.int64())
    ])
    
    df_new = df[['id', 'title', 'embedding', 'original_id']]
    
    db.create_table(table_name, data=df_new, schema=schema)
    print(f"Created table {table_name}")

if __name__ == "__main__":
    main()
