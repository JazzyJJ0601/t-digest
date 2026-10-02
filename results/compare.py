"""t-digest vs the repo's first (naive) version vs an equal-memory random sample.

Error metric: rank error |F(estimate) - q|, where F is the exact empirical CDF of the stream (the usual metric for
quantile sketches), reported in parts per million of the stream. Memory: t-digest stores 2 numbers per centroid
(mean, weight); the random sample (reservoir) is given exactly that many numbers.
"""
import json
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from tdigest import TDigest
from results.naive_digest import NaiveDigest

QS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
N, SEEDS, DELTA = 100_000, 5, 100
DISTS = {"lognormal": lambda r: r.lognormal(0, 1, N), "normal": lambda r: r.standard_normal(N),
         "uniform": lambda r: r.uniform(0, 1, N)}


def rank_err(sorted_data, est, q):
    lo = np.searchsorted(sorted_data, est, side="left") / len(sorted_data)
    hi = np.searchsorted(sorted_data, est, side="right") / len(sorted_data)
    return 0.0 if lo <= q <= hi else min(abs(lo - q), abs(hi - q))


out = {"n": N, "seeds": SEEDS, "compression": DELTA, "quantiles": QS, "dists": {}}
for name, gen in DISTS.items():
    errs = {"t-digest": [], "naive (v1)": [], "random sample": []}
    sizes = {"t-digest": [], "naive (v1)": [], "random sample": []}
    secs = {"t-digest": [], "naive (v1)": []}
    for seed in range(SEEDS):
        data = gen(np.random.default_rng(seed))
        s = np.sort(data)
        t0 = time.time(); td = TDigest(DELTA); td.add_batch(data); n_c = td.centroid_count(); secs["t-digest"].append(time.time() - t0)
        t0 = time.time(); nv = NaiveDigest(DELTA); nv.add_batch(data); secs["naive (v1)"].append(time.time() - t0)
        mem = 2 * n_c
        samp = np.sort(np.random.default_rng(1000 + seed).choice(data, mem, replace=False))
        errs["t-digest"].append([rank_err(s, td.quantile(q), q) for q in QS])
        errs["naive (v1)"].append([rank_err(s, nv.quantile(q), q) for q in QS])
        errs["random sample"].append([rank_err(s, float(np.quantile(samp, q)), q) for q in QS])
        sizes["t-digest"].append(mem); sizes["naive (v1)"].append(2 * len(nv._clusters)); sizes["random sample"].append(mem)
    out["dists"][name] = {k: {"rank_err_ppm": [round(1e6 * float(v), 1) for v in np.mean(errs[k], axis=0)],
                              "numbers_stored": float(np.mean(sizes[k]))} for k in errs}
    for k in secs:
        out["dists"][name][k]["build_seconds"] = round(float(np.mean(secs[k])), 2)
    print(name)
    for k, v in out["dists"][name].items():
        print(f"  {k:14s} stores {v['numbers_stored']:7.0f}  rank err ppm {v['rank_err_ppm']}")
json.dump(out, open("results/compare.json", "w"), indent=1)
