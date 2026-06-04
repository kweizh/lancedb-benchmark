#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import lancedb

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    r = 6371.0 # Earth radius in km
    return r * c

def main():
    parser = argparse.ArgumentParser(description="Geospatial Haversine + Semantic POI Search with LanceDB")
    parser.add_argument("--center-lat", type=float, required=True, help="Query center latitude in degrees")
    parser.add_argument("--center-lon", type=float, required=True, help="Query center longitude in degrees")
    parser.add_argument("--radius-km", type=float, required=True, help="Search radius in kilometres (inclusive)")
    parser.add_argument("--category", type=str, required=True, help="Exact category string to filter on")
    parser.add_argument("--query-vector-path", type=str, required=True, help="Path to JSON file containing query vector")
    parser.add_argument("--top-k", type=int, required=True, help="Maximum number of results to return")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file")
    
    args = parser.parse_args()
    
    # 1. Load the query vector
    if not os.path.exists(args.query_vector_path):
        print(f"Error: Query vector path {args.query_vector_path} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(args.query_vector_path, "r") as f:
            query_vector = json.load(f)
    except Exception as e:
        print(f"Error reading query vector: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not isinstance(query_vector, list) or len(query_vector) != 32:
        print("Error: Query vector must be a length-32 list of floats.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Connect to LanceDB and open the table
    db_path = "/home/user/myproject/lancedb"
    if not os.path.exists(db_path):
        print(f"Error: LanceDB path {db_path} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    try:
        db = lancedb.connect(db_path)
        if "pois" not in db.table_names():
            print("Error: Table 'pois' not found in LanceDB.", file=sys.stderr)
            sys.exit(1)
        tbl = db.open_table("pois")
    except Exception as e:
        print(f"Error connecting to LanceDB or opening table: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Perform vector search to get all items with distances computed
    try:
        # Since we need to filter and rank, we can retrieve all items from the table.
        # We use limit(len(tbl)) to ensure we fetch all rows for filtering.
        num_rows = len(tbl)
        search_results = tbl.search(query_vector).limit(num_rows).to_pandas()
    except Exception as e:
        print(f"Error performing vector search: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Filter results by category and Haversine distance
    filtered_results = []
    for _, row in search_results.iterrows():
        # Exact category match check
        if row["category"] != args.category:
            continue
            
        # Calculate Haversine distance
        dist_km = haversine_distance(args.center_lat, args.center_lon, row["lat"], row["lon"])
        
        # Check if within radius (inclusive)
        if dist_km <= args.radius_km:
            filtered_results.append({
                "id": int(row["id"]),
                "name": str(row["name"]),
                "category": str(row["category"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "distance_km": float(dist_km),
                "vector_distance": float(row["_distance"])
            })
            
    # 5. Sort the filtered results by vector_distance ascending
    filtered_results.sort(key=lambda x: x["vector_distance"])
    
    # 6. Limit to top_k
    final_results = filtered_results[:args.top_k]
    
    # 7. Write output to JSON file
    output_data = {
        "results": final_results
    }
    
    try:
        # Ensure parent directory of output exists
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Successfully wrote {len(final_results)} results to {args.output}")
    sys.exit(0)

if __name__ == "__main__":
    main()
