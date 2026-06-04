import argparse
import json
import lancedb
import numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def main():
    parser = argparse.ArgumentParser(description="Search POIs in LanceDB")
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--radius-km", type=float, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--query-vector-path", type=str, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    # Read query vector
    with open(args.query_vector_path, "r") as f:
        query_vector = json.load(f)
        
    db = lancedb.connect("/home/user/myproject/lancedb")
    table = db.open_table("pois")
    
    # We want to search by vector, filter by category, and get all possible results 
    # (since row count is small, we can just use a large limit)
    # Then filter by radius and take top K.
    # The default distance is L2. `_distance` column is added by LanceDB.
    
    total_rows = table.count_rows()
    escaped_category = args.category.replace("'", "''")
    results_df = table.search(query_vector).where(f"category = '{escaped_category}'").limit(total_rows).to_pandas()
    
    if results_df.empty:
        output_data = {"results": []}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        return
        
    # Calculate Haversine distance
    results_df["distance_km"] = haversine(args.center_lat, args.center_lon, results_df["lat"], results_df["lon"])
    
    # Filter by radius
    filtered_df = results_df[results_df["distance_km"] <= args.radius_km].copy()
    
    # Sort by vector distance
    filtered_df = filtered_df.sort_values(by="_distance", ascending=True)
    
    # Take top K
    top_k_df = filtered_df.head(args.top_k)
    
    # Format output
    output_results = []
    for _, row in top_k_df.iterrows():
        output_results.append({
            "id": int(row["id"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "distance_km": float(row["distance_km"]),
            "vector_distance": float(row["_distance"])
        })
        
    output_data = {"results": output_results}
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()
