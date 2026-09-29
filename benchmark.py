#!/usr/bin/env python3
"""
Benchmarks tdigest quantile error vs exact numpy percentile
on lognormal data.
"""
import numpy as np
import sys

try:
    from tdigest import TDigest
    HAS_TDIGEST = True
except ImportError:
    HAS_TDIGEST = False
    print("tdigest not found; please install it with `pip install tdigest`", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Generate 100k lognormal points
np.random.seed(42)
data = np.random.lognormal(mean=0, sigma=1, size=100_000)

# Quantiles to test
qs = [0.01, 0.5, 0.99, 0.999]

# Build tdigest (add point-by-point for compatibility)
td = TDigest()
for x in data:
    td.add(x)

# Compute exact vs approximate
exact = [np.percentile(data, q * 100) for q in qs]
approx = [td.quantile(q) for q in qs]

# Compute relative error
errors = []
for a, e in zip(approx, exact):
    if e != 0:
        errors.append(abs(a - e) / abs(e))
    else:
        errors.append(abs(a - e))

# Print table
print(f"{'Quantile':<10} {'Exact':<15} {'Approx':<15} {'Rel Error (%)':<15}")
print("-" * 55)
for q, e, a, err in zip(qs, exact, approx, errors):
    print(f"{q:<10.4f} {e:<15.4f} {a:<15.4f} {err*100:<15.4f}")

# Save plot
if HAS_MATPLOTLIB:
    plt.figure(figsize=(6, 4))
    plt.plot(qs, errors, marker='o')
    plt.xlabel('Quantile')
    plt.ylabel('Relative Error')
    plt.title('TDigest vs NumPy Percentile Error')
    plt.savefig('error.png')
    print("\nSaved error.png")
else:
    print("\nSkipping plot (matplotlib not available)")
