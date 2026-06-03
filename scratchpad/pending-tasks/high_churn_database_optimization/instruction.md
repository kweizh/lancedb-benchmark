Unlike fully managed cloud-native databases, LanceDB OSS requires periodic manual optimization to merge small storage fragments and update indexes for newly added data to prevent query latency degradation.

You need to write an automated maintenance script that checks the table's statistics and programmatically triggers optimization routines if the number of unindexed rows exceeds 10,000.

**Constraints:**
- Must programmatically check `num_unindexed_rows` before executing maintenance operations.
- If the threshold is met, you must call both `table.optimize()` and `table.compact_files()`.
- Do not run the optimization commands if the threshold is not reached.