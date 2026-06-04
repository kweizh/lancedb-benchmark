#!/usr/bin/env python3
"""
Geospatial Haversine + Semantic POI Search with LanceDB.

Searches the pre-seeded `pois` table by combining:
  - Geographic filter (Haversine distance <= radius_km)
  - Exact category match
  - Vector similarity ranking (L2, ascending)
"""

import argparse
import json
import math
import sys

import numpy as np
import lancedb


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def l2_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the L2 (Euclidean) distance between two vectors."""
    diff = v1 - v2
    return float(np.dot(diff, diff) ** 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search POIs by geography, category, and vector similarity."
    )
    parser.add_argument("--center-lat", type=float, required=True,
                        help="Query center latitude in degrees.")
    parser.add_argument("--center-lon", type=float, required=True,
                        help="Query center longitude in degrees.")
    parser.add_argument("--radius-km", type=float, required=True,
                        help="Search radius in kilometres (inclusive).")
    parser.add_argument("--category", type=str, required=True,
                        help="Exact category string to filter on.")
    parser.add_argument("--query-vector-path", type=str, required=True,
                        help="Path to a JSON file containing a 32-element float array.")
    parser.add_argument("--top-k", type=int, required=True,
                        help="Maximum number of results to return.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load query vector ---
    with open(args.query_vector_path, "r") as fh:
        query_vector = json.load(fh)
    if not isinstance(query_vector, list) or len(query_vector) != 32:
        print("ERROR: query vector must be a JSON array of 32 floats.", file=sys.stderr)
        sys.exit(1)
    query_vec = np.array(query_vector, dtype=np.float32)

    # --- Connect to LanceDB and load the pois table ---
    db = lancedb.connect("/home/user/myproject/lancedb")
    table = db.open_table("pois")

    # Retrieve all rows as a pandas DataFrame for client-side processing.
    df = table.to_pandas()

    # --- Apply category filter ---
    df = df[df["category"] == args.category].copy()

    if df.empty:
        output = {"results": []}
        with open(args.output, "w") as fh:
            json.dump(output, fh, indent=2)
        sys.exit(0)

    # --- Apply Haversine geographic filter ---
    def _dist(row):
        return haversine_km(args.center_lat, args.center_lon, row["lat"], row["lon"])

    df["distance_km"] = df.apply(_dist, axis=1)
    df = df[df["distance_km"] <= args.radius_km].copy()

    if df.empty:
        output = {"results": []}
        with open(args.output, "w") as fh:
            json.dump(output, fh, indent=2)
        sys.exit(0)

    # --- Compute L2 vector distance ---
    def _vec_dist(row):
        poi_vec = np.array(row["embedding"], dtype=np.float32)
        return l2_distance(poi_vec, query_vec)

    df["vector_distance"] = df.apply(_vec_dist, axis=1)

    # --- Rank by ascending vector_distance, take top-K ---
    df = df.sort_values("vector_distance", ascending=True)
    df = df.head(args.top_k)

    # --- Build output ---
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
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2)

    sys.exit(0)


if __name__ == "__main__":
    main()
