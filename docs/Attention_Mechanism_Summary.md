# Attention Mechanism Summary

This document provides a comparative analysis of the attention mechanisms available in this library. Each mechanism is designed with specific time-series characteristics in mind, offering trade-offs between computational complexity, expressiveness, and inductive bias.

| Mechanism Name | File Location | Complexity | Key Strengths | Key Weaknesses/Limitations | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Attention** | `SelfAttention_Family.py` | O(L²) | Highly expressive, general-purpose. | High computational cost for long sequences. No inherent time-series inductive bias. | Shorter sequences where capturing all pairwise interactions is critical. |
| **Custom MHA** | `CustomMultiHeadAttention.py` | O(L²) | Optimized implementation (Flash Attention), modular, clean API. | Same as Full Attention. | Default choice for standard Transformer blocks on short-to-medium sequences. |
| **ProbAttention** | `SelfAttention_Family.py` | O(L log L) | Efficient approximation of full attention. | Approximation may miss some nuances. Complex implementation. | Long sequences where a balance between efficiency and expressiveness is needed. |
| **Reformer** | `SelfAttention_Family.py` | O(L log L) | Memory-efficient due to LSH, suitable for very long sequences. | Hashing can be less precise than other methods. | Extremely long sequences where memory is the primary constraint. |
| **AutoCorrelation** | `AutoCorrelation.py` | O(L log L) | Strong inductive bias for periodicity, efficient via FFT. | Less effective for non-periodic or aperiodic data. | Time series with clear, dominant periodic patterns. |
| **Enhanced AutoCorrelation** | `EnhancedAutoCorrelation.py` | O(L log L) | Adaptive `k`, multi-scale analysis, learnable filters make it robust and flexible. | Still primarily focused on periodic data. | The preferred choice for most time series, especially those with multiple or complex periodicities. |
| **DSAttention** | `SelfAttention_Family.py` | O(L²) | Adapts to non-stationary data by learning to rescale attention scores. | Adds parameters and complexity. Still has quadratic complexity. | Time series with known or suspected distribution shifts (non-stationarity). |
| **Fourier Attention** | `FourierCorrelation.py` | O(L log L) | Directly models dependencies in the frequency domain. | Can lose time-domain information. Mode selection is a critical hyperparameter. | Signals where frequency-domain analysis is more important than time-domain interactions. |
| **Wavelet Attention** | `MultiWaveletCorrelation.py` | O(L) | Excellent time-frequency localization, captures transient patterns. | Extremely complex implementation, strong modeling assumptions from wavelet choice. | Complex signals like seismic or biomedical data where transient, localized events are critical. |
| **Two-Stage Attention**| `SelfAttention_Family.py` | O(L² + D²) | Models both temporal and inter-feature dependencies explicitly. | High computational cost for high-dimensional series. | Multivariate time series where interactions between different features are as important as temporal patterns. |
