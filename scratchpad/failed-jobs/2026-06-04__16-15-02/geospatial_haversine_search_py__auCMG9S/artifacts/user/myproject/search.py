#!/usr/bin/env python3
"""Geospatial Haversine + Semantic POI Search with LanceDB."""

import argparse
import json
import math

import lancedb
import numpy as np


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km between two points using the Haversine formula."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def main():
    parser = argparse.ArgumentParser(description="Search POIs by geography, category, and vector similarity")
    parser.add_argument("--center-lat", type=float, required=True, help="Query center latitude in degrees")
    parser.add_argument("--center-lon", type=float, required=True, help="Query center longitude in degrees")
    parser.add_argument("--radius-km", type=float, required=True, help="Search radius in kilometres (inclusive)")
    parser.add_argument("--category", type=str, required=True, help="Exact category string to filter on")
    parser.add_argument("--query-vector-path", type=str, required=True, help="Path to JSON file with 32-d query vector")
    parser.add_argument("--top-k", type=int, required=True, help="Maximum number of results to return")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()

    # Load query vector
    with open(args.query_vector_path, "r") as f:
        query_vector = json.load(f)
    query_vector = np.array(query_vector, dtype=np.float32)

    # Connect to LanceDB
    db = lancedb.connect("/home/user/myproject/lancedb")
    tbl = db.open_table("pois")

    # Read all data
    df = tbl.to_pandas()

    # Filter by category
    df = df[df["category"] == args.category].copy()

    # Compute Haversine distance
    df["distance_km"] = haversine_km(
        args.center_lat, args.center_lon,
        df["lat"].values, df["lon"].values,
    )

    # Filter by radius (inclusive)
    df = df[df["distance_km"] <= args.radius_km]

    if len(df) == 0:
        results = []
    else:
        # Compute vector (L2) distance between each POI embedding and the query vector
        embeddings = np.stack(df["embedding"].values)  # shape (N, 32)
        vector_distances = np.linalg.norm(embeddings - query_vector, axis=1)
        df["vector_distance"] = vector_distances

        # Sort by vector_distance ascending (smallest = best match first)
        df = df.sort_values("vector_distance", ascending=True)

        # Take top-K
        df = df.head(args.top_k)

        # Build results
        results = []
        for _, row in df.iterrows():
            results.append({
                "id": int(row["id"]),
                "name": str(row["name"]),
                "category": str(row["category"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "distance_km": float(row["distance_km"]),
                "vector_distance": float(row["vector_distance"]),
            })

    output = {"results": results}

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()