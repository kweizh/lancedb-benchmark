# Embedding Drift Detector

## Background
Production embedding distributions drift over time as upstream data changes (new topics, seasonality, model retrains). You are building an offline drift monitor on top of LanceDB: it compares a frozen **baseline** embedding set against a **current** production embedding set and quantifies the shift with KL and JS divergence over k-means clusters.

Two LanceDB tables have already been seeded at container entrypoint under `/home/user/myproject/data/`:
- `baseline_${ZEALT_RUN_ID}` — 1000 rows of 64-d float32 vectors drawn i.i.d. from N(0, I).
- `current_${ZEALT_RUN_ID}` — 1000 rows of 64-d float32 vectors drawn from a shifted distribution.

Both tables share the schema `{id: int64, vector: fixed_size_list<float32, 64>}`.

## Requirements
Write a Python module `solution.py` exposing a function:

```python
def detect_drift(baseline_table, current_table, n_samples: int = 500) -> dict:
    ...
```

Where `baseline_table` and `current_table` are open LanceDB `Table` objects. The returned dict **MUST** contain exactly these keys:

- `kl_divergence` (float) — KL(current || baseline) in nats over the k-means cluster distribution.
- `js_divergence` (float) — Jensen–Shannon divergence (base e) between the two cluster distributions.
- `drifted` (bool) — True iff `js_divergence > 0.05`.
- `top_shifted_clusters` (list[int]) — the 5 cluster indices with the largest absolute mass-shift `|p_current[c] - p_baseline[c]|`, sorted by absolute shift in **descending** order. Ties are broken by ascending cluster index.

### Algorithm (MUST be implemented exactly as specified for reproducibility)
1. Deterministically sample `n_samples` row indices from each table using `numpy.random.default_rng(2026)` and `rng.choice(total_rows, size=n_samples, replace=False)`. Use one fresh RNG per table (i.e. instantiate `default_rng(2026)` twice — once before sampling the baseline, once before sampling the current).
2. Materialize the sampled vectors as a `(n_samples, 64)` `float32` numpy array per table.
3. Fit `sklearn.cluster.KMeans(n_clusters=20, random_state=42, n_init=10)` on the baseline sample only.
4. Use `.predict(...)` to assign cluster ids to both the baseline sample and the current sample.
5. Build the two 20-bin probability vectors `p_baseline` and `p_current` by counting cluster assignments and dividing by `n_samples`. Add `1e-12` to every bin **before** normalizing to avoid `log(0)` blow-ups when computing divergences.
6. Compute `KL(current || baseline) = sum(p_current * log(p_current / p_baseline))` (natural log).
7. Compute Jensen–Shannon divergence with `M = (p_current + p_baseline) / 2` and `JS = 0.5 * KL(p_current || M) + 0.5 * KL(p_baseline || M)`.
8. `drifted = js_divergence > 0.05`.
9. `top_shifted_clusters = sorted(range(20), key=lambda c: (-abs(p_current[c] - p_baseline[c]), c))[:5]`.

### Driver script
Also provide `run.py` in the same directory that:
1. Reads `run_id = os.environ["ZEALT_RUN_ID"]`.
2. Connects to LanceDB at `./data/`.
3. Opens both `baseline_${run_id}` and `current_${run_id}`.
4. Calls `detect_drift(baseline, current, n_samples=500)` and writes the result to `result.json` next to `run.py`. Floats may be serialized with full precision.

## Implementation Hints
- Use `lancedb.connect("./data")` and `db.open_table(...)`.
- Convert a Lance table to a pandas DataFrame via `tbl.to_pandas()` and stack the `vector` column into a `(N, 64)` array.
- Choose KL / JS implementations that match the spec exactly (numpy + `np.log` is sufficient — no need for `scipy.special.rel_entr` provided the epsilon smoothing is applied as instructed).
- `KMeans.fit` must use `n_init=10` (explicit). Defaults in scikit-learn 1.5 differ.

## Acceptance Criteria
- Project path: `/home/user/myproject`
- Command: `python3 run.py`
- Reads `ZEALT_RUN_ID` from the environment to locate the tables `baseline_${ZEALT_RUN_ID}` and `current_${ZEALT_RUN_ID}` inside `./data/`.
- After a successful run, `/home/user/myproject/result.json` MUST exist and parse as a JSON object with exactly the keys `kl_divergence`, `js_divergence`, `drifted`, `top_shifted_clusters`.
- `result.json["drifted"]` MUST be `true`.
- `result.json["js_divergence"]` MUST be a float strictly greater than `0.05` and strictly less than `1.0`.
- `result.json["kl_divergence"]` MUST be a non-negative float.
- `result.json["top_shifted_clusters"]` MUST be a list of exactly 5 distinct integers in `[0, 20)`, sorted by descending absolute mass-shift (ties broken by ascending cluster id).
- `solution.py` MUST expose `detect_drift(baseline_table, current_table, n_samples=500)` returning a dict with the same schema as above. The verifier will import this function and call it directly against the seeded tables.

