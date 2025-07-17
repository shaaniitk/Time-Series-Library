# ChronosX Integration Guide 🕐

## Required Package Installation

### Method 1: Install Official Chronos Package (Recommended)
```bash
# Install the official Chronos forecasting package
pip install chronos-forecasting

# Optional: Install with specific torch version
pip install chronos-forecasting torch torchvision torchaudio
```

### Method 2: Install from Source (Development)
```bash
# Clone and install from GitHub
git clone https://github.com/amazon-science/chronos-forecasting.git
cd chronos-forecasting
pip install -e .
```

### Method 3: Alternative Installation
```bash
# Install via conda if available
conda install -c conda-forge chronos-forecasting
```

## Current Package Status

### ✅ **Currently Installed:**
- `transformers==4.53.2` - For base HuggingFace support
- `torch` - PyTorch framework
- Standard dependencies (numpy, pandas, etc.)

### ❌ **Missing for ChronosX:**
- `chronos-forecasting` - The official Chronos package
- Proper ChronosPipeline API access

## How to Install ChronosX

### Step 1: Install the Chronos Package
```bash
# Navigate to your project directory
cd "D:\workspace\Time-Series-Library"

# Activate your environment if using one
# source tsl-env/Scripts/activate  # or tsl-env\Scripts\activate on Windows

# Install chronos-forecasting
pip install chronos-forecasting
```

### Step 2: Verify Installation
```python
# Test script to verify Chronos installation
try:
    from chronos import ChronosPipeline
    import torch
    
    # Try loading a small model
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",  # Use CPU for testing
        torch_dtype=torch.float32,
    )
    print("✅ ChronosX successfully installed and working!")
    
    # Test forecasting
    context = torch.randn(1, 24)  # 24 time steps
    forecast = pipeline.predict(context, prediction_length=12)
    print(f"✅ Forecast generated: {forecast.shape}")
    
except ImportError as e:
    print(f"❌ ChronosX not installed: {e}")
    print("📦 Install with: pip install chronos-forecasting")
except Exception as e:
    print(f"⚠️ ChronosX installed but error occurred: {e}")
```

## Available ChronosX Models

### Model Sizes and Specifications
| Model | Parameters | Memory | Speed | Use Case |
|-------|------------|--------|--------|----------|
| `amazon/chronos-t5-tiny` | ~8M | Low | Fast | Development/Testing |
| `amazon/chronos-t5-mini` | ~20M | Low | Fast | Lightweight Production |
| `amazon/chronos-t5-small` | ~60M | Medium | Medium | Standard Use |
| `amazon/chronos-t5-base` | ~200M | Medium | Medium | High Accuracy |
| `amazon/chronos-t5-large` | ~700M | High | Slow | Maximum Performance |

### Model Selection Guide
```python
# For development and testing
model_name = "amazon/chronos-t5-tiny"

# For production with balanced performance
model_name = "amazon/chronos-t5-small"

# For maximum accuracy (requires more resources)
model_name = "amazon/chronos-t5-large"
```

## Updated ChronosX Usage in Modular Architecture

### Proper API Usage
```python
from chronos import ChronosPipeline
import torch

# Load ChronosX model properly
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

# Generate forecasts
context = torch.randn(1, 96)  # Historical data
forecast = pipeline.predict(
    context=context,
    prediction_length=24,
    num_samples=20  # For uncertainty quantification
)
```

### Integration with Modular Architecture
```python
# Using ChronosX in modular configuration
config = ModularConfig(
    backbone_type='chronos_x',          # Uses ChronosPipeline
    processor_type='time_domain',       # Post-processing
    attention_type='multi_head',        # Compatible attention
    loss_type='mse'                     # Standard loss
)

# The ChronosXBackbone will automatically:
# 1. Load ChronosPipeline if available
# 2. Fall back to mock model if not installed
# 3. Provide uncertainty quantification
# 4. Support batch processing
```

## Installation Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Error
```
ImportError: No module named 'chronos'
```
**Solution:**
```bash
pip install chronos-forecasting
# or
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

#### Issue 2: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
```python
# Use smaller model or CPU
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",  # Smaller model
    device_map="cpu",          # Use CPU
    torch_dtype=torch.float32  # Use float32 instead of bfloat16
)
```

#### Issue 3: Dependency Conflicts
```
Package conflicts with transformers version
```
**Solution:**
```bash
# Update transformers first
pip install --upgrade transformers
pip install chronos-forecasting
```

## Testing ChronosX Integration

### Step 1: Install ChronosX
```bash
pip install chronos-forecasting
```

### Step 2: Run Integration Test
```bash
python test_chronos_x_simple.py
```

### Step 3: Expected Output
```
🚀 Testing ChronosX Integration with Modular Architecture
📦 Available ChronosX Backbones:
  ✅ chronos_x: ChronosX-based backbone with HF Transformers
  ✅ chronos_x_tiny: Tiny ChronosX model for fast experimentation
  ✅ chronos_x_large: Large ChronosX model for maximum performance
  ✅ chronos_x_uncertainty: ChronosX optimized for uncertainty quantification
```

## Mock vs Real ChronosX

### Current Behavior (Without chronos-forecasting)
- ✅ **Architecture works** - Modular system operational
- ⚠️ **Mock models used** - Fallback implementation
- ✅ **Testing possible** - Can test component combinations
- ❌ **No real forecasting** - Mock predictions only

### After Installing chronos-forecasting
- ✅ **Real ChronosX models** - Actual Amazon Chronos models
- ✅ **Proper forecasting** - High-quality predictions
- ✅ **Uncertainty quantification** - True probabilistic forecasts
- ✅ **Production ready** - Real-world deployment

## Next Steps

1. **Install ChronosX**: `pip install chronos-forecasting`
2. **Test Integration**: Run the test scripts
3. **Verify Forecasting**: Check real vs mock predictions
4. **Production Deployment**: Use real ChronosX models

The modular architecture is **already working** with mock models, so installing ChronosX will seamlessly upgrade it to use real models! 🚀
