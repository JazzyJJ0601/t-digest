# t-digest-from-scratch Design

This repository implements a t-digest streaming quantile sketch from scratch using pure Python and NumPy. The goal is to efficiently estimate quantiles in data streams while supporting mergeable summaries, enabling distributed computation.

## Modules
- `core.py`: Defines the TDigest class and centroid data structure.
- `clustering.py`: Implements the merge logic for combining two digests efficiently.
- `benchmarks.py`: Runs accuracy tests against NumPy baselines.

## Public API
- `TDigest`: Main class representing the sketch.
- `add(value)`: Incorporates a new data point into the digest.
- `get_quantile(q)`: Returns the estimated quantile value for q in [0, 1].
- `merge(other)`: Combines this digest with another TDigest instance, preserving accuracy guarantees.

## Benchmark Plan
We will generate synthetic heavy-tailed data using a Pareto distribution. For various sample sizes (10^4 to 10^6), we will compare the t-digest quantile estimates against exact values computed by numpy.quantile. Metrics include absolute error at key quantiles (0.5, 0.9, 0.99) and memory footprint. This validates accuracy under realistic streaming conditions where full data retention is impossible. We will also test scalability by measuring update and query throughput.

Tests will cover edge cases like empty inputs, single elements, and repeated merges to ensure robustness.
