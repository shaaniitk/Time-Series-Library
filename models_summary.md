# Time Series Library - Models Overview

## Introduction

This document provides a comprehensive overview of the models implemented in the Time Series Library. The library contains a wide range of state-of-the-art time series forecasting models, from traditional transformer-based architectures to specialized models designed for specific time series tasks.

## Base Architecture

The library follows a modular design with a clear hierarchy of base classes:

- `BaseTimeSeriesModel`: The abstract base class for all time series models, providing a unified interface for model initialization and forward passes.
- `ModelConfig`: A dataclass that standardizes configuration across different model types.
- `InHouseModelBase`: Extends the base model for in-house implementations.

This architecture ensures consistency across different model implementations and simplifies the process of adding new models to the library.

## Model Categories

The models in the library can be categorized into several groups:

### Transformer-Based Models

- **Transformer**: The standard transformer architecture adapted for time series forecasting.
- **Informer**: An efficient transformer variant with a ProbSparse self-attention mechanism.
- **Autoformer**: A decomposition-based transformer with auto-correlation mechanisms.
- **FEDformer**: Frequency Enhanced Decomposed Transformer with efficient frequency domain analysis.
- **Pyraformer**: A pyramid attention-based transformer for time series forecasting.
- **Reformer**: An efficient transformer variant using locality-sensitive hashing for attention.
- **iTransformer**: An innovative transformer architecture designed specifically for time series data.

### Linear Models

- **DLinear**: A simple yet effective decomposition-based linear model.
- **TiDE**: Temporal Decomposition and Embedding model with linear layers.

### Specialized Models

- **PatchTST**: A model that processes time series data in patches for better feature extraction.
- **TimesNet**: A model that captures both time and frequency domain information.
- **ETSformer**: A transformer-based model inspired by Exponential Smoothing (ETS).
- **Koopa**: A specialized architecture for time series forecasting with enhanced pattern recognition.
- **SCINet**: A model that captures multi-scale temporal dependencies.
- **MICN**: A model with multi-scale information processing capabilities.
- **FiLM**: Feature-wise Linear Modulation for time series forecasting.
- **TimeMixer**: A model that mixes information across different time scales.
- **TSMixer**: A simple MLP-based mixer architecture for time series.
- **LightTS**: A lightweight model for efficient time series forecasting.
- **FreTS**: A frequency-enhanced time series model.
- **WPMixer**: A wavelet-based patch mixer for time series forecasting.

### Recurrent Neural Network Models

- **SegRNN**: A segmented recurrent neural network for time series forecasting.

### Mamba-Based Models

- **Mamba**: An implementation of the Mamba architecture for time series forecasting.
- **MambaSimple**: A simplified version of the Mamba architecture.
- **MambaHierarchical**: A hierarchical version of Mamba for multi-level time series.

### Enhanced and Specialized Variants

The library also includes numerous enhanced and specialized variants of the base models:

- **BayesianEnhancedAutoformer**: Adds Bayesian capabilities to the Autoformer model.
- **HierarchicalEnhancedAutoformer**: Designed for hierarchical time series forecasting.
- **QuantileBayesianAutoformer**: Supports quantile forecasting with Bayesian methods.
- **EnhancedAutoformer**: An improved version of the Autoformer with additional capabilities.

### HuggingFace Integration

The library includes models with HuggingFace integration, indicated by the "HF" prefix:

- **HFAutoformerSuite**: A suite of Autoformer models integrated with HuggingFace.
- **HFBayesianAutoformer**: Bayesian Autoformer with HuggingFace integration.
- **HFEnhancedAutoformer**: Enhanced Autoformer with HuggingFace integration.
- **HFHierarchicalAutoformer**: Hierarchical Autoformer with HuggingFace integration.
- **HFQuantileAutoformer**: Quantile forecasting Autoformer with HuggingFace integration.

## Model Factory and Unified Architecture

The library implements a factory pattern for model creation:

- **unified_autoformer_factory.py**: Provides a unified factory for creating different Autoformer variants.
- **HFAdvancedFactory.py**: An advanced factory for HuggingFace-integrated models.

## Task Support

The models in the library support various time series tasks:

1. **Long-term Forecasting**: Predicting values far into the future.
2. **Short-term Forecasting**: Predicting values in the near future.
3. **Imputation**: Filling in missing values in time series data.
4. **Anomaly Detection**: Identifying unusual patterns in time series data.
5. **Classification**: Classifying time series into different categories.

## Model Implementation Example: Autoformer

The Autoformer model is a key implementation in the library. It features:

- Series decomposition to separate trend and seasonal components
- Auto-correlation mechanisms instead of traditional self-attention
- An encoder-decoder architecture with specialized layers
- Support for multiple time series tasks

The model's forward method handles different tasks by calling specialized methods:

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
    if self.task_name == 'imputation':
        dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out
    if self.task_name == 'anomaly_detection':
        dec_out = self.anomaly_detection(x_enc)
        return dec_out
    if self.task_name == 'classification':
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out
```

## Conclusion

The models folder in the Time Series Library provides a comprehensive collection of state-of-the-art time series forecasting models. The modular design and unified interfaces make it easy to use existing models and add new ones. The library supports various time series tasks and provides specialized models for different use cases.