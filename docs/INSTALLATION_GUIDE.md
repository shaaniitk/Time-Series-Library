# ChronosX Enhanced Time Series Library - Installation Guide

This guide provides comprehensive installation instructions for the enhanced Time Series Library with ChronosX integration and modular architecture.

## System Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for larger models)
- **Storage**: 10GB+ free space for models and data
- **OS**: Windows, Linux, or macOS

## Installation Options

### Option 1: Full Installation (Recommended)

Install all dependencies including ChronosX, testing suite, and development tools:

```bash
# Clone the repository
git clone <repository-url>
cd Time-Series-Library

# Install all requirements
pip install -r requirements.txt

# Verify ChronosX installation
python -c "from chronos import ChronosPipeline; print('ChronosX installed successfully')"
```

### Option 2: Minimal Installation

For basic functionality without ChronosX integration:

```bash
# Install core dependencies only
pip install torch numpy pandas matplotlib scikit-learn
pip install einops reformer-pytorch local-attention
pip install PyWavelets statsmodels sympy tqdm
```

### Option 3: Development Installation

For contributors and developers:

```bash
# Full installation plus development tools
pip install -r requirements.txt

# Install additional development dependencies
pip install pre-commit isort autoflake
pre-commit install
```

## GPU Support (Optional but Recommended)

### CUDA 11.8 (Recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU Only
```bash
pip install torch torchvision torchaudio
```

## Key New Dependencies Explained

### ChronosX Integration
- **chronos-forecasting**: Amazon's pre-trained time series models
- **transformers**: Hugging Face transformer models support
- **accelerate**: Efficient model loading and inference
- **datasets**: Data handling for pre-trained models

### Enhanced Testing and Benchmarking
- **seaborn**: Advanced statistical visualizations
- **plotly**: Interactive plotting capabilities
- **psutil**: System resource monitoring
- **pytest**: Comprehensive testing framework

### Advanced Time Series Models
- **xgboost**, **lightgbm**, **catboost**: Gradient boosting frameworks
- **pmdarima**: ARIMA model automation
- **prophet**: Facebook's forecasting model
- **mxnet**: Additional deep learning framework

### Development and Analysis
- **jupyter**: Interactive development environment
- **black**, **flake8**, **mypy**: Code quality tools
- **hydra-core**: Configuration management
- **polars**, **dask**: High-performance data processing

## Verification Steps

After installation, verify your setup:

```bash
# Test core functionality
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test ChronosX integration
python -c "from chronos import ChronosPipeline; print('✅ ChronosX ready')"

# Test modular components
python -c "from layers.modular.core.registry import unified_registry; print('✅ Modular components ready')"

# Run quick test
python demo_models/chronos_x_simple_demo.py --smoke
```

## Configuration for Different Use Cases

### 1. Research and Development
```bash
# Full installation with all features
pip install -r requirements.txt
```

### 2. Production Deployment
```bash
# Core dependencies + ChronosX + monitoring
pip install torch numpy pandas matplotlib scikit-learn
pip install chronos-forecasting transformers accelerate
pip install psutil seaborn
```

### 3. Educational Use
```bash
# Basic installation + Jupyter
pip install torch numpy pandas matplotlib scikit-learn
pip install jupyter ipywidgets seaborn
pip install chronos-forecasting
```

## Memory and Performance Optimization

### For Limited Memory Systems (< 8GB)
```python
# Use smaller ChronosX models
config.model_name = "amazon/chronos-t5-tiny"
config.num_samples = 5
config.batch_size = 1
```

### For High Performance Systems (16GB+)
```python
# Use larger models for better accuracy
config.model_name = "amazon/chronos-t5-base"
config.num_samples = 50
config.batch_size = 8
```

## Troubleshooting Common Issues

### 1. ChronosX Installation Fails
```bash
# Update pip and try again
pip install --upgrade pip setuptools wheel
pip install chronos-forecasting --no-cache-dir
```

### 2. Memory Issues During Model Loading
```bash
# Reduce model size or use CPU
export CUDA_VISIBLE_DEVICES=""
python your_script.py
```

### 3. Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

### 4. CUDA Issues
```bash
# Check CUDA version
nvcc --version

# Reinstall appropriate PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Features Enabled by New Dependencies

### 1. ChronosX Integration (`chronos-forecasting`)
- Pre-trained Amazon Chronos models (tiny, small, base, large)
- Zero-shot time series forecasting
- Uncertainty quantification
- Production-ready inference

### 2. Enhanced Visualizations (`seaborn`, `plotly`)
- Statistical distribution plots
- Interactive time series charts
- Model performance comparisons
- Real-time monitoring dashboards

### 3. Advanced Testing (`pytest`, `psutil`)
- Comprehensive test suites
- Performance benchmarking
- Resource usage monitoring
- Stress testing capabilities

### 4. Development Tools (`black`, `mypy`, `jupyter`)
- Code formatting and linting
- Type checking
- Interactive development
- Documentation generation

### 5. Configuration Management (`hydra-core`, `omegaconf`)
- Structured configuration files
- Experiment management
- Parameter sweeps
- Environment-specific configs

## Quick Start Examples

### Basic ChronosX Usage
```python
from chronos import ChronosPipeline
import torch
import numpy as np

# Load model
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")

# Generate forecast
context = torch.randn(1, 100)
forecast = pipeline.predict(context, prediction_length=24)
```

### Modular Architecture Usage
```python
from models.modular_autoformer import ModularAutoformer
from layers.modular.core.registry import unified_registry

# Initialize with ChronosX backbone
configs.use_backbone_component = True
configs.backbone_type = "chronos_tiny"

model = ModularAutoformer(configs)
```

### Production Testing
```python
from chronos_x_production_testing import ChronosXProductionTester

tester = ChronosXProductionTester()
results = tester.run_production_testing_suite()
```

## Support and Documentation

- **ChronosX Documentation**: https://github.com/amazon-science/chronos-forecasting
- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **Project Documentation**: See `/docs` directory
- **Examples**: See test files and demo scripts

## Version Compatibility

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|-------------------|--------|
| Python | 3.8 | 3.9+ | Better performance with 3.9+ |
| PyTorch | 2.1.0 | 2.1.0+ | Required for ChronosX |
| chronos-forecasting | 1.5.2 | Latest | Core ChronosX functionality |
| transformers | 4.30.0 | Latest | HuggingFace models |
| numpy | <2.0 | 1.24+ | Compatibility requirement |

---

**Note**: This installation guide covers the enhanced Time Series Library with ChronosX integration, modular architecture, and comprehensive testing capabilities. All dependencies in `requirements.txt` are tested and verified to work together.
