# Geospatial Haversine + Semantic POI Search with LanceDB

## Background
A pre-seeded LanceDB table contains a corpus of Points-of-Interest (POIs) with geographic coordinates, a category label, and a precomputed 32-dimensional embedding for each POI. You must build a small Python CLI that searches the POI corpus by a combination of geographic distance, category filter, and vector similarity.

The table is pre-seeded at container build time from a deterministic NumPy generator. The candidate code must not regenerate the table contents — it must read from the table that already exists on disk.

## Requirements
- Implement a Python 3 command line program at `/home/user/myproject/search.py`.
- The program must connect to the pre-seeded LanceDB database under `/home/user/myproject/lancedb`.
- The program must read the existing table `pois` (which contains the columns `id INT32`, `name STRING`, `lat FLOAT64`, `lon FLOAT64`, `category STRING`, `embedding FIXED_SIZE_LIST<FLOAT32, 32>`).
- Given a query center `(lat, lon)`, a radius in kilometres, a category filter, a query embedding (32-d), and a top-K integer, the program must return the top-K POIs that satisfy ALL of the following, ranked by L2 distance between the POI embedding and the query embedding (smallest distance first):
  1. The great-circle distance between the POI coordinates and the query center (Haversine formula, Earth radius = 6371.0 km) is less than or equal to the radius in kilometres.
  2. The POI `category` exactly matches the provided category filter.
- The program must write its output to a JSON file at the path passed via `--output`.

## Implementation Hints
- The implementation choice for the geographic filter is up to you. You may materialise a Haversine distance column on the LanceDB side using `add_columns({...})` with a SQL expression and then filter via `where(...)`, or you may run the vector search first and post-filter in pandas/NumPy, or you may compute everything client-side after reading the table. Any approach is acceptable as long as the final ranked output is correct.
- The Earth radius constant must be `6371.0` km exactly to match the reference implementation.
- LanceDB's default vector distance is L2; you may use `.search(query_vector)` directly without specifying a distance type.
- Use `argparse` (or any other argument parser) — the CLI signature is fixed (see Acceptance Criteria).
- The query embedding file is a plain JSON file whose top-level value is an array of 32 floats, e.g. `[0.12, -0.34, ...]`.

## Acceptance Criteria
- Project path: /home/user/myproject
- Command: `python3 /home/user/myproject/search.py`
- Command-line arguments (all required):
  - `--center-lat <float>`: the query center latitude in degrees.
  - `--center-lon <float>`: the query center longitude in degrees.
  - `--radius-km <float>`: the search radius in kilometres (inclusive).
  - `--category <str>`: exact category string to filter on.
  - `--query-vector-path <path>`: path to a JSON file containing a length-32 array of floats.
  - `--top-k <int>`: the maximum number of results to return.
  - `--output <path>`: path to the output JSON file.
- Database location: `/home/user/myproject/lancedb` (LanceDB connection string).
- Table name: `pois`.
- Output JSON schema written to `--output`:
  ```json
  {
    "results": [
      {
        "id": <int>,
        "name": <string>,
        "category": <string>,
        "lat": <float>,
        "lon": <float>,
        "distance_km": <float>,
        "vector_distance": <float>
      }
    ]
  }
  ```
  - The `results` array MUST be ordered by ascending `vector_distance` (smallest = best match first).
  - The array length MUST be `min(top_k, number_of_matching_pois)`.
  - All POIs in `results` MUST satisfy the radius and category constraints.
- The program MUST exit with status 0 on success.

