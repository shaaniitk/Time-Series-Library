# Time Series Library - Utils Overview

## Introduction

This document provides a comprehensive overview of the utility modules in the Time Series Library. The utils folder contains essential components that support the models, including data processing, loss functions, metrics, time feature extraction, and modular component architecture.

## Core Utilities

### Loss Functions

The `losses.py` module provides various loss functions for time series forecasting:

- **Standard Losses**: MSE, MAE, RMSE, etc.
- **Time Series Specific Losses**: MAPE, SMAPE, MASE
- **Advanced Losses**: PSLoss (Patch-wise Structural Loss) that combines point-wise loss with structural losses calculated over patches of time series

The PSLoss implementation includes a sophisticated Fourier-based Adaptive Patching (FAP) algorithm that identifies meaningful segments in time series data based on dominant frequencies.

### Metrics

The `metrics.py` module provides evaluation metrics for time series forecasting:

- **RSE**: Root Squared Error
- **CORR**: Correlation coefficient
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **MSPE**: Mean Squared Percentage Error

These metrics are used to evaluate model performance on various time series tasks.

### Time Features

The `timefeatures.py` module provides time-based feature extraction:

- **Time Feature Classes**: SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, MonthOfYear, WeekOfYear
- **Frequency-based Feature Selection**: Automatically selects appropriate time features based on the data frequency

These features are normalized to the range [-0.5, 0.5] and provide important temporal context for the models.

### Model Utilities

The `model_utils.py` module provides utilities for model operations:

- **Gradient Clipping**: Prevents exploding gradients during training
- **Parameter Counting**: Tracks model size and complexity
- **Weight Initialization**: Properly initializes model weights
- **Input Validation**: Ensures model inputs meet expected requirements
- **Model Profiling**: Monitors performance and memory usage

## Modular Component Architecture

The `modular_components` folder implements a sophisticated modular architecture that allows for flexible model composition:

### Base Interfaces

The `base_interfaces.py` module defines abstract base classes for all modular components:

- **BaseComponent**: The fundamental interface for all components
- **BaseBackbone**: For backbone models (Chronos, T5, etc.)
- **BaseEmbedding**: For embedding components that convert input data into model representation
- **BaseAttention**: For attention mechanisms (self-attention, cross-attention, etc.)
- **BaseProcessor**: For processing strategies (seq2seq, encoder-only, etc.)
- **BaseFeedForward**: For feed-forward networks
- **BaseOutput**: For output layers
- **BaseLoss**: For loss computation components
- **BaseAdapter**: For adapter components that connect different parts of the model

These interfaces ensure compatibility and interchangeability across different model architectures.

### Component Factory

The `factory.py` module implements a factory pattern for creating modular components:

- **ComponentFactory**: Creates components with proper type checking and configuration validation
- **Type Mapping**: Maps component types to their base interfaces
- **Error Handling**: Provides clear error messages for invalid component types or configurations

### Component Registry

The `registry.py` module maintains a registry of available components:

- **ComponentRegistry**: Registers and retrieves component implementations
- **Auto-registration**: Components can be automatically registered using decorators

### Component Implementations

The `implementations` folder contains concrete implementations of the base interfaces:

- **Adapters**: Connect different parts of the model
- **Attentions**: Various attention mechanisms
- **Backbones**: Core model architectures
- **Embeddings**: Input embedding strategies
- **FeedForward**: Feed-forward network implementations
- **Losses**: Loss function implementations
- **Outputs**: Output layer implementations
- **Processors**: Processing strategy implementations

## Additional Utilities

### Data Analysis

The `data_analysis.py` module provides tools for analyzing time series data:

- **Statistical Analysis**: Computes statistics of time series data
- **Visualization**: Generates plots for data exploration

### Augmentation

The `augmentation.py` module provides data augmentation techniques for time series:

- **Time Warping**: Stretches or compresses time series
- **Magnitude Warping**: Scales the magnitude of time series
- **Jittering**: Adds noise to time series

### Masking

The `masking.py` module provides masking utilities for time series:

- **Random Masking**: Randomly masks time series values
- **Structured Masking**: Masks time series values in a structured way

### Scaler Management

The `scaler_manager.py` module provides scaling utilities for time series:

- **StandardScaler**: Standardizes time series data
- **MinMaxScaler**: Scales time series data to a specific range
- **RobustScaler**: Scales time series data robustly to outliers

### Bayesian Extensions

The `bayesian_losses.py` module provides Bayesian loss functions:

- **KL Divergence**: Measures the difference between probability distributions
- **Evidence Lower Bound (ELBO)**: Used for variational inference

### Quantile Extensions

The `quantile_utils.py` module provides utilities for quantile forecasting:

- **Quantile Loss**: Computes loss for quantile forecasting
- **Quantile Evaluation**: Evaluates quantile forecasts

## Conclusion

The utils folder in the Time Series Library provides a comprehensive set of utilities that support the models. The modular component architecture allows for flexible model composition, while the various utility modules provide essential functionality for time series forecasting. The well-designed interfaces and factory pattern make it easy to extend the library with new components and functionality.