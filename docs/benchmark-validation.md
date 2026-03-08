# Search performance validation (Stage 9)

Use this process to validate search latency on larger repositories and confirm the 500ms target:

1. Build hashbrown with CGO enabled.
2. Run the benchmark suite below for 1K, 10K, and 100K LOC synthetic reference repos.
3. Confirm the reported steady-state search average (`ns/op`) stays near or under 500ms. Measure tail latency separately if you need a p95 budget.

```bash
CGO_ENABLED=1 go test -run TestNonExistent -bench BenchmarkSearchLatencyAgainstRepoSizes ./internal/search -count=1
```

The benchmark harness generates temporary repos containing synthetic Go code at the requested LOC scales, indexes them with the internal indexer and synthetic embedder, then repeatedly executes semantic search.

## Recorded results

| LOC scale | Benchmark target | Result | Notes |
|-----------|-----------------|--------|-------|
| 1K        | `BenchmarkSearchLatencyAgainstRepoSizes/1k-loc-14` | 359,577 ns/op | Warm machine, synthetic repo, single query path |
| 10K       | `BenchmarkSearchLatencyAgainstRepoSizes/10k-loc-14` | 350,710 ns/op | Warm machine, synthetic repo, single query path |
| 100K      | `BenchmarkSearchLatencyAgainstRepoSizes/100k-loc-14` | 369,004 ns/op | Warm machine, synthetic repo, single query path |

Re-run this command on each target environment before release and paste fresh numbers into this table.
