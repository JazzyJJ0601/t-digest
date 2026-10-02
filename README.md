# t-digest

Accurate percentiles of a huge stream (p50, p99, p99.9 latency and the like) from a summary of about 60 numbers, in plain Python and NumPy.
This is the merging t-digest of Dunning & Ertl (2019), written from scratch.

## The idea

You can't keep every value of a stream, but you can keep a short sorted list of centroids (a mean and a count).
The trick is how big each centroid may grow. The k₁ scale function

k(q) = δ/(2π) · asin(2q − 1)

lets a centroid span at most one unit of k. Near the median k is flat, so one centroid can hold thousands of points; near q = 0 and q = 1 it is steep, so centroids hold only a few. The tails, where p99 and p99.9 live, stay sharp while the summary stays tiny. New points are buffered and merged in sorted order, and two digests merge the same way.

## Result

`python results/compare.py` → `results/compare.json`. 100,000 points, 5 seeds, compression δ = 100.
Error is rank error: how far the estimate's true position in the stream is from the quantile asked for, in parts per million (100 ppm = 0.01%).

**Lognormal (heavy right tail)**

| Method | Numbers stored | p0.1 | p1 | p50 | p99 | p99.9 |
|---|---|---|---|---|---|---|
| **t-digest (this repo)** | **119** | **58** | **104** | **534** | 504 | 312 |
| First version of this repo | 6,589 | 266 | 618 | 638 | 350 | 260 |
| Random sample, same memory | 119 | 7,502 | 11,368 | 50,018 | 4,134 | 3,114 |

On normal and uniform data the picture is the same (all rows in `results/compare.json`): the t-digest stays between 120 and 550 ppm at every quantile, while a random sample of equal size is 5,000 to 68,000 ppm off.

- Against a random sample of the same size: 10 to 270 times smaller error, depending on distribution and quantile.
- Against this repo's first version (nearest-cluster merging with a flat 100-point cap, kept in `results/naive_digest.py`): about the same error with 55 times less memory, and 120 times faster to build (0.07 s vs 8.5 s for 100,000 points).

## Limits

- Rank error is the right yardstick for a quantile sketch, but in value terms a sparse tail magnifies it: on the lognormal stream the p99.9 estimate is 0.04% off in rank and 13% off in value (24.5 vs 21.7), because only 100 points lie beyond it. `python benchmark.py` prints both. More compression (δ = 200 or more) or treating single-point centroids as exact would tighten this.
- Pure Python adds; fine for millions of points, not for billions.

## Use

```python
from tdigest import TDigest

td = TDigest(compression=100)
td.add_batch(latencies_ms)
td.quantile(0.99)        # p99
td.cdf(250.0)            # fraction of requests at or under 250 ms
td.merge(other_digest)   # combine digests from different machines
```

```bash
pip install -e . && pip install pytest
pytest -q tests
python benchmark.py
python results/compare.py
```

## Reference

Dunning, T. & Ertl, O. (2019). Computing Extremely Accurate Quantiles Using t-Digests. arXiv:1902.04023.
