import numpy as np

class TDigest:
    def __init__(self, compression=100):
        self.compression = compression
        self._clusters = []  # list of [mean, weight]
        self._total_weight = 0

    def add(self, x):
        self._total_weight += 1

        if not self._clusters:
            self._clusters.append([float(x), 1])
            return

        # Find cluster with mean closest to x
        best_i = 0
        best_dist = abs(self._clusters[0][0] - x)
        for i in range(1, len(self._clusters)):
            d = abs(self._clusters[i][0] - x)
            if d < best_dist:
                best_dist = d
                best_i = i

        # Merge if cluster weight is below compression limit
        if self._clusters[best_i][1] < self.compression:
            mean, w = self._clusters[best_i]
            new_w = w + 1
            self._clusters[best_i][0] = (mean * w + x) / new_w
            self._clusters[best_i][1] = new_w
            return

        # Create new cluster
        self._clusters.append([float(x), 1])

    def add_batch(self, values):
        """Add multiple values at once."""
        for x in values:
            self.add(x)

    def merge(self, other):
        for mean, w in other._clusters:
            for _ in range(w):
                self.add(mean)

    def quantile(self, q):
        if not self._clusters or self._total_weight == 0:
            return float('nan')
        
        # Sort clusters by mean for proper quantile estimation
        sorted_clusters = sorted(self._clusters, key=lambda c: c[0])
        
        target = q * self._total_weight
        cum = 0.0
        for i, (mean, w) in enumerate(sorted_clusters):
            if cum + w >= target:
                if w == 0:
                    continue
                return float(mean)
            cum += w
        return float(sorted_clusters[-1][0])

    def cdf(self, x):
        if not self._clusters:
            return 0.0
        cum = 0.0
        for mean, w in self._clusters:
            if mean <= x:
                cum += w
        return cum / self._total_weight if self._total_weight > 0 else 0.0
