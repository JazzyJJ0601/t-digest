# Spectrum Adapter (spectrum-adapter)

A lightweight library for adaptive Fourier feature selection in neural networks.

## What is Spectrum Adapter?

Spectrum Adapter is a neural network component that learns to select the most informative Fourier frequencies for a given regression task. Traditional MLPs struggle with high-frequency targets due to spectral bias. This library addresses that by wrapping standard linear layers with adaptive Fourier embeddings, allowing the model to focus computation on relevant frequency bands.

## Core Idea

Instead of manually designing feature maps, Spectrum Adapter:
1. Starts with a bank of sinusoidal basis functions at different frequencies
2. Learns to assign importance weights to each frequency
3. Prunes irrelevant frequencies during training for efficiency

This combines the benefits of Fourier feature networks with automatic feature selection.

## API Sketch

```python
from spectrum_adapter import SpectrumAdapter

# Create adapter for 10-dimensional input
adapter = SpectrumAdapter(input_dim=10, max_freqs=64)

# Training interface
adapter.fit(X_train, y_train, epochs=100, lr=0.001)

# Predictions
y_pred = adapter.predict(X_test)

# Inspect learned frequencies
freq_importance = adapter.get_importance()
```

## Planned Tests

1. **Sine Wave Reconstruction**: Fit a sum of 5 sinusoids at known frequencies; verify adapter learns correct importance weights.
2. **Noise Robustness**: Add Gaussian noise to targets; check that adapter suppresses high-frequency noise and generalizes better than standard MLP.
3. **Efficiency Benchmark**: Compare memory footprint and inference latency against a full-bank Fourier network with 256 frequencies.

## References

- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions", NeurIPS 2020
- Zhang et al., "Adaptive Spectral Representation", ICML 2021
