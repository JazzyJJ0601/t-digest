"""Merging t-digest (Dunning & Ertl, "Computing extremely accurate quantiles using t-digests", 2019).

Points are buffered, then merged into a sorted list of centroids (mean, weight). The k1 scale function
k(q) = delta / (2*pi) * asin(2q - 1) limits each centroid to one unit of k, so centroids near q = 0 and
q = 1 hold very few points (accurate tails) and centroids near the median hold many (compact).
"""
import math

import numpy as np


def _k(q, delta):
    return delta / (2 * math.pi) * math.asin(2 * min(max(q, 0.0), 1.0) - 1)


class TDigest:
    def __init__(self, compression=100, buffer_size=None):
        self.compression = float(compression)
        self._means = np.empty(0)
        self._weights = np.empty(0)
        self._buffer = []
        self._buffer_size = buffer_size or int(5 * compression)
        self._total_weight = 0.0
        self._min = math.inf
        self._max = -math.inf

    # ---- building ---------------------------------------------------------
    def add(self, x, w=1.0):
        x = float(x)
        self._buffer.append((x, float(w)))
        self._min, self._max = min(self._min, x), max(self._max, x)
        if len(self._buffer) >= self._buffer_size:
            self._compress()

    def add_batch(self, values):
        for x in values:
            self.add(x)

    def merge(self, other):
        """Fold another digest's centroids in (weights kept), so the result summarises both streams."""
        other._compress()
        for m, w in zip(other._means, other._weights):
            self._buffer.append((float(m), float(w)))
        self._min, self._max = min(self._min, other._min), max(self._max, other._max)
        self._compress()

    def _compress(self):
        if not self._buffer:
            return
        bm, bw = zip(*self._buffer)
        means = np.concatenate([self._means, bm])
        weights = np.concatenate([self._weights, bw])
        self._buffer = []
        order = np.argsort(means, kind="mergesort")
        means, weights = means[order], weights[order]
        total = weights.sum()
        out_m, out_w = [means[0]], [weights[0]]
        q_left = 0.0                       # quantile at the left edge of the current centroid
        k_left = _k(q_left, self.compression)
        for m, w in zip(means[1:], weights[1:]):
            q_right = q_left + (out_w[-1] + w) / total
            if _k(q_right, self.compression) - k_left <= 1.0:
                nw = out_w[-1] + w
                out_m[-1] += (m - out_m[-1]) * w / nw
                out_w[-1] = nw
            else:
                q_left += out_w[-1] / total
                k_left = _k(q_left, self.compression)
                out_m.append(m)
                out_w.append(w)
        self._means, self._weights = np.array(out_m), np.array(out_w)
        self._total_weight = float(total)

    # ---- queries ----------------------------------------------------------
    def centroid_count(self):
        self._compress()
        return len(self._means)

    def quantile(self, q):
        """Estimated value at quantile q in [0, 1], interpolating between centroid centres."""
        self._compress()
        n = len(self._means)
        if n == 0:
            return float("nan")
        if n == 1:
            return float(self._means[0])
        total = self._total_weight
        target = q * total
        # centroid i is centred at cumulative weight c[i]
        c = np.cumsum(self._weights) - self._weights / 2
        if target <= c[0]:
            # between the minimum (weight 0) and the first centre
            return float(self._min + (self._means[0] - self._min) * (target / c[0] if c[0] > 0 else 1.0))
        if target >= c[-1]:
            span = total - c[-1]
            return float(self._means[-1] + (self._max - self._means[-1]) * ((target - c[-1]) / span if span > 0 else 0.0))
        i = int(np.searchsorted(c, target, side="right")) - 1
        t = (target - c[i]) / (c[i + 1] - c[i])
        return float(self._means[i] + t * (self._means[i + 1] - self._means[i]))

    def cdf(self, x):
        """Estimated fraction of points <= x."""
        self._compress()
        n = len(self._means)
        if n == 0:
            return 0.0
        if x < self._min:
            return 0.0
        if x >= self._max:
            return 1.0
        total = self._total_weight
        c = np.cumsum(self._weights) - self._weights / 2
        if x <= self._means[0]:
            span = self._means[0] - self._min
            return float(c[0] * ((x - self._min) / span if span > 0 else 1.0) / total)
        if x >= self._means[-1]:
            span = self._max - self._means[-1]
            return float((c[-1] + (total - c[-1]) * ((x - self._means[-1]) / span if span > 0 else 0.0)) / total)
        i = int(np.searchsorted(self._means, x, side="right")) - 1
        span = self._means[i + 1] - self._means[i]
        t = (x - self._means[i]) / span if span > 0 else 0.0
        return float((c[i] + t * (c[i + 1] - c[i])) / total)
