#!/usr/bin/env python3
"""Personalized recommender that blends user taste with a query vector."""

import argparse
import json
import os

import lancedb
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Personalized recommender")
    parser.add_argument("--user-id", required=True, help="User identifier")
    parser.add_argument("--query-vec", required=True, help="Path to .npy query vector")
    parser.add_argument("--k", type=int, required=True, help="Number of results")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    # Read blending coefficient from environment
    alpha = float(os.environ.get("BLEND_ALPHA", "0.3"))

    # Connect to LanceDB
    db = lancedb.connect("/home/user/project/data")
    items_table = db.open_table("items")
    history_table = db.open_table("user_history")

    # Load query vector
    query_vec = np.load(args.query_vec).astype(np.float32)

    # Load up to 10 most recent interactions for the user
    history_df = (
        history_table.to_pandas()
        .query(f"user_id == '{args.user_id}'")
        .sort_values("ts", ascending=False)
        .head(10)
    )

    if len(history_df) == 0:
        # No history: use pure query vector
        blended = query_vec
        seen_ids = []
    else:
        seen_ids = history_df["item_id"].tolist()

        # Retrieve embeddings for interacted items
        items_df = items_table.to_pandas()
        item_embeddings = items_df[items_df["id"].isin(seen_ids)]["vector"]

        # Compute user taste vector as mean of item embeddings
        taste_vec = np.mean(np.stack(item_embeddings.values), axis=0).astype(
            np.float32
        )

        # Blend: (1 - alpha) * query + alpha * taste
        blended = ((1 - alpha) * query_vec + alpha * taste_vec).astype(np.float32)

    # Build the search query
    search = items_table.search(blended.tolist())

    # Exclude already-seen items
    if seen_ids:
        # Build a NOT IN filter using SQL syntax
        ids_str = ", ".join(str(i) for i in seen_ids)
        search = search.where(f"id NOT IN ({ids_str})")

    # Run search and take top-k
    results = search.limit(args.k).to_pandas()

    # Extract ranked item IDs
    result_ids = results["id"].astype(int).tolist()

    # Write output
    with open(args.output, "w") as f:
        json.dump(result_ids, f)


if __name__ == "__main__":
    main()