import os
import time
import lancedb
import numpy as np

def build_index():
    uri = os.environ.get("LANCEDB_URI", "/app/lancedb")
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    db = lancedb.connect(uri)
    table_name = f"vectors_{run_id}"
    tbl = db.open_table(table_name)
    
    # Build IVF_PQ index on vector column
    tbl.create_index(
        metric="l2",
        num_partitions=64,
        num_sub_vectors=8,
        vector_column_name="vector",
        replace=True,
        index_type="IVF_PQ"
    )
    
    # Wait for index to finish building
    indices = tbl.list_indices()
    index_names = [idx.name for idx in indices]
    if "vector_idx" not in index_names:
        index_names.append("vector_idx")
    tbl.wait_for_index(index_names)

def sweep(query_set, param_grid, k=10):
    uri = os.environ.get("LANCEDB_URI", "/app/lancedb")
    run_id = os.environ.get("ZEALT_RUN_ID", "")
    if not run_id:
        raise ValueError("ZEALT_RUN_ID environment variable is not set")
    
    db = lancedb.connect(uri)
    table_name = f"vectors_{run_id}"
    tbl = db.open_table(table_name)
    
    # 1. Compute ground-truth exact top-k for each query
    # Using brute-force (index-bypassing) scan
    exact_top_k_sets = []
    for q in query_set:
        # Fetch more than k to handle potential tie-breaking deterministically
        limit_val = max(k * 2, 100)
        exact_results = tbl.search(q).bypass_vector_index().limit(limit_val).to_list()
        
        # Break ties by ascending id
        exact_results.sort(key=lambda x: (x["_distance"], x["id"]))
        exact_top_k = [x["id"] for x in exact_results[:k]]
        exact_top_k_sets.append(set(exact_top_k))
        
    nprobes_list = param_grid.get("nprobes", [])
    refine_factor_list = param_grid.get("refine_factor", [])
    
    results = []
    
    # Evaluate the full Cartesian product, sorted ascending by (nprobes, refine_factor)
    for nprobes in sorted(nprobes_list):
        for refine_factor in sorted(refine_factor_list):
            recalls = []
            latencies = []
            
            for i, q in enumerate(query_set):
                t0 = time.perf_counter()
                ann_results = (
                    tbl.search(q)
                    .nprobes(nprobes)
                    .refine_factor(refine_factor)
                    .limit(k)
                    .to_list()
                )
                t1 = time.perf_counter()
                
                latencies.append((t1 - t0) * 1000.0)
                
                # Extract IDs from ANN search
                ann_results_sorted = sorted(ann_results, key=lambda x: (x["_distance"], x["id"]))
                ann_top_k = [x["id"] for x in ann_results_sorted]
                
                # Compute recall@k
                intersection = len(set(ann_top_k) & exact_top_k_sets[i])
                recall = intersection / k
                recalls.append(recall)
                
            mean_recall = sum(recalls) / len(recalls)
            mean_latency = sum(latencies) / len(latencies)
            
            results.append({
                "nprobes": int(nprobes),
                "refine_factor": int(refine_factor),
                "recall": float(mean_recall),
                "mean_latency_ms": float(mean_latency)
            })
            
    # The list must be sorted ascending by (nprobes, refine_factor)
    results.sort(key=lambda x: (x["nprobes"], x["refine_factor"]))
    return results
