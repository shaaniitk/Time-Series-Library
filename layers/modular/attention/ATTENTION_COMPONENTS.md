# Modular Attention Components Documentation

This document provides a mathematical and implementation overview of all attention components in the modular architecture (`layers/modular/attention`).

## 1. FourierAttention
- **Purpose:** Captures periodic patterns using frequency domain analysis.
- **Mathematical Principle:** Applies Fast Fourier Transform (FFT) to input sequences, learns frequency filters, and reconstructs attention weights via inverse FFT.
- **Key Equation:**
  $$\text{Attention}(X) = \text{IFFT}(W_f \cdot \text{FFT}(X))$$
  where $W_f$ are learnable frequency weights.

## 2. AdaptiveAutoCorrelationAttention
- **Purpose:** Enhances autocorrelation with adaptive window selection.
- **Mathematical Principle:** Computes autocorrelation over multiple scales, adaptively selects top-k correlations.
- **Key Equation:**
  $$\text{AutoCorr}(X) = \max_k \left( \sum_{i=1}^n X_i X_{i+k} \right)$$

## 3. WaveletAttention / MultiScaleWaveletAttention / AdaptiveWaveletAttention
- **Purpose:** Multi-resolution analysis using wavelet transforms.
- **Mathematical Principle:** Decomposes input into wavelet coefficients at multiple scales, computes attention in wavelet domain.
- **Key Equation:**
  $$X = \sum_{j=1}^J W_j \psi_j$$
  where $W_j$ are wavelet coefficients, $\psi_j$ are wavelet basis functions.

## 4. TemporalConvAttention / CausalConvolution / TemporalConvNet
- **Purpose:** Sequence modeling via causal/dilated convolutions.
- **Mathematical Principle:** Applies convolutional filters with dilation to capture long-range dependencies.
- **Key Equation:**
  $$Y_t = \sum_{k=0}^{K-1} w_k X_{t-dk}$$
  where $d$ is dilation, $w_k$ are filter weights.

## 5. BayesianAttention
- **Purpose:** Models uncertainty in attention weights.
- **Mathematical Principle:** Treats attention weights as random variables, samples from learned distributions.
- **Key Equation:**
  $$A \sim \mathcal{N}(\mu, \sigma^2)$$

## 6. CrossResolutionAttention
- **Purpose:** Integrates information across multiple temporal resolutions.
- **Mathematical Principle:** Computes attention weights for each resolution and aggregates.
- **Key Equation:**
  $$A = \sum_{r=1}^R \alpha_r A_r$$
  where $A_r$ is attention at resolution $r$, $\alpha_r$ are learnable aggregation weights.

## 7. MetaLearningAdapter
- **Purpose:** Fast adaptation to new patterns via meta-learning.
- **Mathematical Principle:** Learns adaptation weights using gradient-based updates.
- **Key Equation:**
  $$\theta' = \theta - \alpha \nabla_\theta L(\theta)$$

## Missing Components (To Be Implemented)
- **MultiWaveletCrossAttention:** Cross-attention using multi-wavelet transforms.
- **TwoStageAttention:** Two-stage attention for segment merging and cross-dimension/time.
- **ExponentialSmoothingAttention:** Attention based on exponential smoothing weights.

## Implementation Notes
- All attention modules inherit from `BaseAttention`.
- Each module supports configurable parameters (e.g., `d_model`, `n_heads`, `levels`).
- Mathematical operations are implemented using PyTorch and NumPy.

## References
- See `layers/modular/attention/*.py` for source code.
- For further details, refer to the integration report and architecture documentation.
