import os
import numpy as np

# Set environment variables for testing
os.environ["LANCEDB_URI"] = "/app/lancedb"
os.environ["ZEALT_RUN_ID"] = "test_run"

from solution import build_index, sweep

def main():
    print("Building index...")
    build_index()
    print("Index built successfully!")

    # Generate some deterministic query vectors
    # We can use the same seed to generate deterministic queries
    rng = np.random.default_rng(42)
    query_set = rng.standard_normal((10, 64)).astype("float32")

    param_grid = {
        "nprobes": [1, 4, 16, 64],
        "refine_factor": [1, 2, 4, 10]
    }

    print("Running sweep run 1...")
    results1 = sweep(query_set, param_grid, k=10)

    print("\nResults Run 1:")
    print(f"{'nprobes':<10} | {'refine_factor':<15} | {'recall':<10} | {'mean_latency_ms':<20}")
    print("-" * 65)
    for r in results1:
        print(f"{r['nprobes']:<10} | {r['refine_factor']:<15} | {r['recall']:<10.4f} | {r['mean_latency_ms']:<20.4f}")

    print("\nRunning sweep run 2 (to test determinism)...")
    results2 = sweep(query_set, param_grid, k=10)

    # Check determinism
    for r1, r2 in zip(results1, results2):
        assert r1["nprobes"] == r2["nprobes"]
        assert r1["refine_factor"] == r2["refine_factor"]
        assert r1["recall"] == r2["recall"], f"Recall mismatch: {r1['recall']} vs {r2['recall']}"
    print("\nDeterminism check passed!")

    # Check non-decreasing recall with effort
    # Let's check that as nprobes or refine_factor increases, recall does not decrease.
    # Specifically, check that the highest-effort configuration has higher recall than the lowest-effort one.
    lowest_effort = results1[0]
    highest_effort = results1[-1]
    print(f"\nLowest effort recall: {lowest_effort['recall']:.4f}")
    print(f"Highest effort recall: {highest_effort['recall']:.4f}")
    assert highest_effort["recall"] >= lowest_effort["recall"], "Recall decreased with effort!"
    assert highest_effort["recall"] > lowest_effort["recall"], "Recall did not increase with highest effort!"
    print("Recall improvement check passed!")

if __name__ == "__main__":
    main()
