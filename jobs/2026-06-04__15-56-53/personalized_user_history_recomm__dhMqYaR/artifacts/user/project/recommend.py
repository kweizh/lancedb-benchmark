#!/usr/bin/env python3
"""
Personalized recommender that blends a user's taste vector (derived from
recent interaction history) with a query vector and runs a LanceDB vector
search against the items catalogue.
"""

import argparse
import json
import os

import numpy as np
import lancedb

DB_PATH = "/home/user/project/data"
ITEMS_TABLE = "items"
HISTORY_TABLE = "user_history"
DEFAULT_ALPHA = 0.3
MAX_HISTORY = 10


def load_user_history(db: lancedb.DBConnection, user_id: str) -> list[int]:
    """Return the up-to-MAX_HISTORY most-recent item IDs for *user_id*."""
    history_tbl = db.open_table(HISTORY_TABLE)
    df = (
        history_tbl.search()
        .where(f"user_id = '{user_id}'", prefilter=True)
        .to_pandas()
    )
    if df.empty:
        return []
    # Sort by timestamp descending, keep the 10 most recent
    df = df.sort_values("ts", ascending=False).head(MAX_HISTORY)
    return df["item_id"].tolist()


def get_item_embeddings(db: lancedb.DBConnection, item_ids: list[int]) -> np.ndarray:
    """Fetch embeddings for the given item IDs and return them as an (N, D) array."""
    items_tbl = db.open_table(ITEMS_TABLE)
    id_list = ", ".join(str(i) for i in item_ids)
    df = (
        items_tbl.search()
        .where(f"id IN ({id_list})", prefilter=True)
        .to_pandas()
    )
    # Preserve the order of item_ids so the mean is consistent
    df = df.set_index("id").loc[item_ids]
    vectors = np.stack(df["vector"].values)  # shape (N, 64)
    return vectors


def blend_vectors(
    query_vec: np.ndarray, taste_vec: np.ndarray, alpha: float
) -> np.ndarray:
    """Return (1 - alpha) * query + alpha * taste, normalised to float32."""
    blended = (1.0 - alpha) * query_vec + alpha * taste_vec
    return blended.astype(np.float32)


def run_search(
    db: lancedb.DBConnection,
    query_vec: np.ndarray,
    seen_ids: list[int],
    k: int,
) -> list[int]:
    """
    Vector-search the items table with *query_vec*, excluding *seen_ids*.
    Returns the top-k item IDs in rank order.
    """
    items_tbl = db.open_table(ITEMS_TABLE)

    search_q = items_tbl.search(query_vec.tolist())

    if seen_ids:
        id_list = ", ".join(str(i) for i in seen_ids)
        search_q = search_q.where(f"id NOT IN ({id_list})", prefilter=True)

    # Retrieve more candidates than k to account for any edge-case filtering,
    # then take the top-k from the ranked results.
    results_df = search_q.limit(k).to_pandas()
    return results_df["id"].tolist()[:k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personalised vector-search recommender using LanceDB."
    )
    parser.add_argument("--user-id", required=True, help="User identifier string.")
    parser.add_argument(
        "--query-vec",
        required=True,
        help="Path to a .npy file containing the 64-d query vector.",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of recommendations to return."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file that will receive the ranked item IDs.",
    )
    args = parser.parse_args()

    # --- blending coefficient ---
    try:
        alpha = float(os.environ.get("BLEND_ALPHA", DEFAULT_ALPHA))
    except ValueError:
        raise SystemExit(
            f"BLEND_ALPHA must be a float, got: {os.environ['BLEND_ALPHA']!r}"
        )

    # --- load query vector ---
    query_vec: np.ndarray = np.load(args.query_vec).astype(np.float32).flatten()

    # --- connect to LanceDB ---
    db = lancedb.connect(DB_PATH)

    # --- load user history ---
    seen_ids = load_user_history(db, args.user_id)

    # --- compute taste vector and blend ---
    if seen_ids:
        embeddings = get_item_embeddings(db, seen_ids)
        taste_vec = embeddings.mean(axis=0).astype(np.float32)
        final_vec = blend_vectors(query_vec, taste_vec, alpha)
    else:
        # No history — fall back to pure query vector
        final_vec = query_vec

    # --- vector search with exclusion of seen items ---
    top_k_ids = run_search(db, final_vec, seen_ids, args.k)

    # --- write output ---
    with open(args.output, "w") as fh:
        json.dump([int(i) for i in top_k_ids], fh)

    print(f"Wrote {len(top_k_ids)} recommendations to {args.output}")


if __name__ == "__main__":
    main()
