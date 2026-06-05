import lancedb
import numpy as np
from sklearn.decomposition import PCA
import os
import pyarrow as pa

# Configuration
db_path = "/home/user/myproject/lancedb/"
source_table_name = "articles"
target_run_id = os.environ.get("ZEALT_RUN_ID")
target_table_name = f"articles_pca_{target_run_id}"
pca_model_path = "/app/pca_model.npz"

def main():
    # 1. Read data from source LanceDB table
    db = lancedb.connect(db_path)
    table = db.open_table(source_table_name)
    df = table.to_pandas()
    
    print(f"Read {len(df)} rows from {source_table_name}")
    
    # Extract embeddings
    embeddings = np.stack(df['embedding'].values)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 2. Fit PCA model
    pca = PCA(n_components=16, random_state=42)
    projected_embeddings = pca.fit_transform(embeddings)
    print(f"Projected embeddings shape: {projected_embeddings.shape}")
    
    # 3. Persist PCA model
    # Ensure /app exists (though the environment should have it)
    os.makedirs("/app", exist_ok=True)
    np.savez(pca_model_path, components=pca.components_, mean=pca.mean_)
    print(f"Saved PCA model to {pca_model_path}")
    
    # 4. Create new table with projected embeddings
    # Prepare data for new table
    new_df = df[['id', 'title']].copy()
    new_df['original_id'] = df['id']
    # Convert projected embeddings to list of float32
    new_df['embedding'] = [v.astype(np.float32).tolist() for v in projected_embeddings]
    
    # Define schema for the new table
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 16)),
        pa.field("original_id", pa.int64())
    ])
    
    # Create target table
    if target_table_name in db.table_names():
        db.drop_table(target_table_name)
    
    db.create_table(target_table_name, data=new_df, schema=schema)
    print(f"Created table {target_table_name} with {len(new_df)} rows")

if __name__ == "__main__":
    main()
