import argparse
import json
import os
import numpy as np
import lancedb

def main():
    parser = argparse.ArgumentParser(description="Personalized User-History Recommender")
    parser.add_argument("--user-id", type=str, required=True, help="ID of the user")
    parser.add_argument("--query-vec", type=str, required=True, help="Path to the query vector .npy file")
    parser.add_argument("--k", type=int, required=True, help="Number of recommendations to return")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    # The LanceDB database lives at /home/user/project/data
    db_path = "/home/user/project/data"
    db = lancedb.connect(db_path)
    
    # Open tables
    items_table = db.open_table("items")
    history_table = db.open_table("user_history")

    # 1. Load up to 10 most recent interactions for the user
    # Also load all interacted items for exclusion later
    # Using quotes for user_id in case it's a string
    history_df = history_table.search().where(f"user_id = '{args.user_id}'").to_pandas()
    
    all_seen_ids = []
    recent_item_ids = []
    
    if not history_df.empty:
        all_seen_ids = history_df["item_id"].unique().tolist()
        # Sort by timestamp (ts) descending to get most recent
        recent_history = history_df.sort_values("ts", ascending=False).head(10)
        recent_item_ids = recent_history["item_id"].tolist()

    # 2. Compute user taste vector as the mean of item embeddings from the history
    taste_vector = None
    if recent_item_ids:
        # Retrieve embeddings for the recent items from the items table
        id_list_str = ",".join(map(str, recent_item_ids))
        recent_items_df = items_table.search().where(f"id IN ({id_list_str})").to_pandas()
        
        if not recent_items_df.empty:
            # Extract vectors and compute mean
            # LanceDB vector column is usually returned as a list or numpy array in pandas
            vectors = np.stack(recent_items_df["vector"].values)
            taste_vector = np.mean(vectors, axis=0)

    # 3. Load query vector from .npy file
    try:
        query_vector = np.load(args.query_vec)
    except Exception as e:
        print(f"Error loading query vector: {e}")
        return

    # 4. Blend taste vector with current query vector
    # Blending coefficient from BLEND_ALPHA environment variable (default 0.3)
    alpha = float(os.environ.get("BLEND_ALPHA", 0.3))
    
    if taste_vector is not None:
        # Formula: (1 - alpha) * query + alpha * taste
        blended_vector = (1 - alpha) * query_vector + alpha * taste_vector
    else:
        # Fallback to pure query vector if no history
        blended_vector = query_vector

    # 5. Run vector search on items table and exclude already-seen items
    search_query = items_table.search(blended_vector)
    
    if all_seen_ids:
        # Exclude items the user has already interacted with
        seen_ids_str = ",".join(map(str, all_seen_ids))
        search_query = search_query.where(f"id NOT IN ({seen_ids_str})")
    
    # Execute search and get top-k
    results_df = search_query.limit(args.k).to_pandas()

    # 6. Write top-k item IDs as a JSON array to output file
    if not results_df.empty:
        output_ids = results_df["id"].astype(int).tolist()
    else:
        output_ids = []
    
    # Ensure the output contains exactly k items if available, or fewer if not
    # results_df.head(args.k) is already handled by .limit(args.k)
    
    with open(args.output, "w") as f:
        json.dump(output_ids, f)

if __name__ == "__main__":
    main()
