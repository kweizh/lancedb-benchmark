#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import lancedb

def main():
    parser = argparse.ArgumentParser(description="Personalized user-history recommender with LanceDB")
    parser.add_argument("--user-id", required=True, help="User ID to get recommendations for")
    parser.add_argument("--query-vec", required=True, help="Path to the .npy file containing the query vector")
    parser.add_argument("--k", required=True, type=int, help="Number of items to recommend")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    # Load BLEND_ALPHA from environment variable, defaulting to 0.3
    alpha_str = os.environ.get("BLEND_ALPHA", "0.3")
    try:
        alpha = float(alpha_str)
    except ValueError:
        alpha = 0.3

    # Load query vector
    if not os.path.exists(args.query_vec):
        raise FileNotFoundError(f"Query vector file not found: {args.query_vec}")
    query_vector = np.load(args.query_vec)
    query_vector = np.array(query_vector, dtype=np.float32)

    # Connect to LanceDB
    db_path = "/home/user/project/data"
    db = lancedb.connect(db_path)
    
    items_table = db.open_table("items")
    user_history_table = db.open_table("user_history")

    # Get all history for the user
    history_df = user_history_table.search().where(f"user_id = '{args.user_id}'").to_pandas()

    blended_vector = query_vector
    all_interacted_ids = []

    if len(history_df) > 0:
        all_interacted_ids = history_df['item_id'].unique().tolist()
        
        # Get up to 10 most recent interactions for the taste vector
        recent_df = history_df.sort_values(by="ts", ascending=False).head(10)
        recent_item_ids = recent_df['item_id'].unique().tolist()
        
        if recent_item_ids:
            id_list_str = ",".join(map(str, recent_item_ids))
            recent_items_df = items_table.search().where(f"id IN ({id_list_str})").to_pandas()
            
            if len(recent_items_df) > 0:
                vectors = np.stack(recent_items_df['vector'].values)
                taste_vector = np.mean(vectors, axis=0)
                blended_vector = (1.0 - alpha) * query_vector + alpha * taste_vector

    # Search items table against the blended vector
    search_query = items_table.search(blended_vector)
    
    # Exclude items the user has already interacted with
    if all_interacted_ids:
        id_list_str = ",".join(map(str, all_interacted_ids))
        search_query = search_query.where(f"id NOT IN ({id_list_str})")
        
    res_df = search_query.limit(args.k).to_pandas()
    
    # Extract item IDs in rank order (best match first)
    recommended_ids = res_df['id'].tolist()
    recommended_ids = [int(x) for x in recommended_ids]
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Write to output file
    with open(args.output, "w") as f:
        json.dump(recommended_ids, f)

if __name__ == "__main__":
    main()
