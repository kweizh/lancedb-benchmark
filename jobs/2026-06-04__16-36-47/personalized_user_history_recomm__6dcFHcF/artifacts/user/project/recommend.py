import argparse
import os
import json
import numpy as np
import lancedb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--query-vec", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    alpha = float(os.environ.get("BLEND_ALPHA", 0.3))
    
    db = lancedb.connect("/home/user/project/data")
    items_table = db.open_table("items")
    history_table = db.open_table("user_history")

    # Get all history for the user
    history_df = history_table.search().where(f"user_id = {args.user_id}").to_pandas()
    
    taste_vector = None
    all_seen_item_ids = []
    
    if not history_df.empty:
        all_seen_item_ids = history_df["item_id"].tolist()
        
        # Sort by ts descending and take top 10
        recent_df = history_df.sort_values(by="ts", ascending=False).head(10)
        recent_item_ids = recent_df["item_id"].tolist()
        
        if recent_item_ids:
            recent_items_str = ", ".join(map(str, recent_item_ids))
            recent_items_df = items_table.search().where(f"id IN ({recent_items_str})").to_pandas()
            
            if not recent_items_df.empty:
                # Calculate mean vector
                vectors = np.stack(recent_items_df["vector"].values)
                taste_vector = np.mean(vectors, axis=0)

    # Read query vector
    query_vector = np.load(args.query_vec)
    
    if taste_vector is not None:
        blended_vector = (1 - alpha) * query_vector + alpha * taste_vector
    else:
        blended_vector = query_vector
        
    # Run vector search
    search = items_table.search(blended_vector).limit(args.k)
    
    # Exclude all seen items
    if all_seen_item_ids:
        seen_str = ", ".join(map(str, all_seen_item_ids))
        search = search.where(f"id NOT IN ({seen_str})")
        
    results_df = search.to_pandas()
    
    # Write top k item IDs
    result_ids = results_df["id"].tolist()
    
    with open(args.output, "w") as f:
        json.dump(result_ids, f)

if __name__ == "__main__":
    main()
