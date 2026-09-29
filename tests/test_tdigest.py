import numpy as np
import pytest
from tdigest.digest import TDigest


def test_quantile_accuracy_uniform():
    """Quantile accuracy on 100k uniform samples (rank error < 1% at q=0.5, 0.99)."""
    np.random.seed(42)
    n = 100000
    data = np.random.uniform(0, 1, n)
    
    td = TDigest(compression=100)
    for x in data:
        td.add(x)
    
    # For uniform(0,1), true quantile value = q
    # Rank error: |(rank(value)/n) - q| < 0.01
    for q in [0.5, 0.99]:
        estimated = td.quantile(q)
        # Find actual rank of estimated value
        rank = np.sum(data <= estimated) / n
        error = abs(rank - q)
        assert error < 0.01, f"Uniform: q={q}, rank error={error:.4f}"


def test_quantile_accuracy_normal():
    """Quantile accuracy on 100k normal samples (rank error < 1% at q=0.5, 0.99)."""
    np.random.seed(42)
    n = 100000
    data = np.random.randn(n)  # Standard normal
    
    td = TDigest(compression=100)
    for x in data:
        td.add(x)
    
    for q in [0.5, 0.99]:
        estimated = td.quantile(q)
        rank = np.sum(data <= estimated) / n
        error = abs(rank - q)
        assert error < 0.01, f"Normal: q={q}, rank error={error:.4f}"


def test_merge():
    """Merge of two digests should produce equivalent result to adding all data together."""
    np.random.seed(123)
    data1 = np.random.randn(50000)
    data2 = np.random.randn(50000)
    
    td1 = TDigest(compression=100)
    for x in data1:
        td1.add(x)
    
    td2 = TDigest(compression=100)
    for x in data2:
        td2.add(x)
    
    td_merged = TDigest(compression=100)
    for x in data1:
        td_merged.add(x)
    for x in data2:
        td_merged.add(x)
    
    td1.merge(td2)
    
    for q in [0.1, 0.5, 0.9, 0.99]:
        est1 = td1.quantile(q)
        est_merged = td_merged.quantile(q)
        # Allow some tolerance for numerical differences
        assert abs(est1 - est_merged) < 0.1, f"Merge q={q}: {est1:.4f} vs {est_merged:.4f}"


def test_monotonic_quantiles():
    """Quantiles should be monotonic (non-decreasing)."""
    np.random.seed(456)
    data = np.random.randn(10000)
    
    td = TDigest(compression=100)
    for x in data:
        td.add(x)
    
    qs = np.linspace(0, 1, 100)
    quantiles = [td.quantile(q) for q in qs]
    
    for i in range(1, len(quantiles)):
        assert quantiles[i] >= quantiles[i-1], f"Monotonicity violated at q={qs[i]}"


def test_empty_digest():
    """Empty digest should return NaN."""
    td = TDigest(compression=100)
    
    result = td.quantile(0.5)
    assert np.isnan(result), f"Empty digest should return NaN, got {result}"


def test_single_element():
    """Single element digest should return that element."""
    td = TDigest(compression=100)
    td.add(42.0)
    
    assert td.quantile(0.0) == 42.0
    assert td.quantile(0.5) == 42.0
    assert td.quantile(1.0) == 42.0
