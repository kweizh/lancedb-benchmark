import os
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.decomposition import PCA

def main():
    # 1. Connect to the source LanceDB
    db_path = "/home/user/myproject/lancedb/"
    db = lancedb.connect(db_path)
    
    tbl = db.open_table("articles")
    df = tbl.to_pandas()
    print(f"Loaded {len(df)} rows from source table 'articles'.")
    
    # 2. Extract embeddings
    embeddings = np.stack(df['embedding'].values).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 3. Fit PCA
    pca = PCA(n_components=16, random_state=42, svd_solver='full')
    projected_embeddings = pca.fit_transform(embeddings).astype(np.float32)
    print(f"Projected embeddings shape: {projected_embeddings.shape}")
    
    # 4. Save model to /app/pca_model.npz
    components = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    print(f"Components shape: {components.shape}, mean shape: {mean.shape}")
    
    np.savez("/app/pca_model.npz", components=components, mean=mean)
    print("Saved PCA model to /app/pca_model.npz")
    
    # 5. Create new table articles_pca_${ZEALT_RUN_ID}
    zealt_run_id = os.environ.get("ZEALT_RUN_ID")
    if not zealt_run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set!")
    
    new_table_name = f"articles_pca_{zealt_run_id}"
    print(f"Creating new table: {new_table_name}")
    
    # Prepare the data for the new table
    # Schema columns: id (int), title (str), embedding (fixed-size list of 16 float32), original_id (int)
    new_data = {
        "id": df["id"].values.astype(np.int64),
        "title": df["title"].values,
        "embedding": [list(x) for x in projected_embeddings],
        "original_id": df["id"].values.astype(np.int64)
    }
    
    new_df = pd.DataFrame(new_data)
    
    # Define PyArrow schema to guarantee fixed_size_list
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 16)),
        pa.field("original_id", pa.int64())
    ])
    
    # If table already exists, we should delete it or raise an error.
    # The requirement says "A new LanceDB table named ... exists ...".
    # Let's drop it if it exists, or just create it.
    if new_table_name in db.table_names():
        print(f"Table {new_table_name} already exists. Dropping it first.")
        db.drop_table(new_table_name)
        
    new_tbl = db.create_table(new_table_name, data=new_df, schema=schema)
    print(f"Created table {new_table_name} with {len(new_tbl)} rows.")
    print("Schema of the new table:")
    print(new_tbl.schema)

if __name__ == "__main__":
    main()
