# Time Series Library - Test Organization

This document explains the test organization and how to run tests in the Time Series Library.

## 📁 Test Folder Structure

```
tests/
├── chronosx/                    # ChronosX integration tests
│   ├── test_chronosx_simple.py
│   ├── test_modular_autoformer_chronosx.py
│   ├── test_chronos_x_model_sizes.py
│   ├── chronos_x_benchmark_suite.py
│   └── demo_chronosx_modular.py
├── modular_framework/           # Modular component tests
├── core_algorithms/             # Core algorithm tests
├── integration/                 # Integration tests
├── unit/                        # Unit tests
├── models/                      # Model-specific tests
└── training_validation/         # Training and validation tests
```

## 🚀 Running Tests

### Quick Start
```bash
# Run quick validation suite
python run_all_tests.py quick

# List all available test suites
python run_all_tests.py list

# Run ChronosX tests only
python run_all_tests.py category --target chronosx

# Run specific test suite
python run_all_tests.py suite --target chronosx_core
```

### Test Categories

| Category | Description | Priority | Est. Time |
|----------|-------------|----------|-----------|
| **chronosx** | ChronosX integration and backbone tests | High | ~15min |
| **modular** | Modular framework and component tests | High | ~5min |
| **models** | Enhanced model variants and HF integration | High | ~10min |
| **algorithms** | Core time series algorithms | High | ~3min |
| **integration** | End-to-end workflow tests | Medium | ~5min |
| **performance** | Performance benchmarks (GPU recommended) | Low | ~10min |
| **unit** | Individual component unit tests | High | ~2min |

### Available Test Suites

#### 🔮 ChronosX Tests
- `chronosx_core` - Core ChronosX integration with ModularAutoformer
- `chronosx_models` - Different model sizes and configurations
- `chronosx_comprehensive` - Comprehensive testing and benchmarks
- `chronosx_production` - Production deployment tests
- `chronosx_demos` - Interactive demos and examples

#### 🧠 Modular Framework Tests
- `modular_core` - Core modular component system
- `modular_integration` - Component integration tests

#### 🎯 Model Tests
- `enhanced_models` - Enhanced Autoformer variants
- `bayesian_models` - Bayesian models with uncertainty
- `hf_models` - HuggingFace integration tests

#### 🔧 Algorithm Tests
- `core_algorithms` - Core time series algorithms
- `unit_tests` - Individual component tests
- `integration` - End-to-end workflows
- `performance` - Performance benchmarks

## 📋 Test Commands

### Basic Usage
```bash
# List all test suites
python run_all_tests.py list

# Run quick validation (recommended first)
python run_all_tests.py quick

# Run ChronosX tests only
python run_all_tests.py category -t chronosx

# Run all tests (takes ~30+ minutes)
python run_all_tests.py all
```

### Advanced Usage
```bash
# Run with verbose output
python run_all_tests.py suite -t chronosx_core -v

# Generate detailed report
python run_all_tests.py category -t chronosx -r chronosx_report.json

# Filter by category when listing
python run_all_tests.py list -c chronosx
```

## ✅ Recommended Test Sequence

For new installations or after code changes:

1. **Quick Validation** (2 minutes)
   ```bash
   python run_all_tests.py quick
   ```

2. **ChronosX Integration** (15 minutes)
   ```bash
   python run_all_tests.py category -t chronosx
   ```

3. **Core Components** (10 minutes)
   ```bash
   python run_all_tests.py category -t modular
   python run_all_tests.py category -t models
   ```

4. **Full Suite** (optional, 30+ minutes)
   ```bash
   python run_all_tests.py all
   ```

## 🐛 Troubleshooting

### Common Issues

**ChronosX Tests Failing:**
- Check installation: `python verify_installation.py`
- Ensure sufficient memory (8GB+ recommended)
- Verify internet connection for model downloads

**Module Import Errors:**
- Run from project root directory
- Check Python path in tests
- Verify all dependencies installed

**Timeout Errors:**
- Tests have 2x estimated time as timeout
- Reduce model sizes for faster testing
- Check system resources

### Test Dependencies

**Required for all tests:**
- Python 3.8+
- PyTorch
- Basic dependencies (numpy, pandas, etc.)

**ChronosX tests:**
- chronos-forecasting package
- transformers
- 8GB+ RAM recommended

**Performance tests:**
- GPU recommended
- Additional memory
- Extended time allocation

## 📊 Test Reports

Test reports include:
- ✅ Pass/fail status for each test
- ⏱️ Execution times
- 📋 Detailed error messages
- 📈 Performance metrics
- 🎯 Coverage information

Example report generation:
```bash
python run_all_tests.py category -t chronosx -r chronosx_report.json
```

## 🔧 Adding New Tests

To add new tests:

1. **Choose appropriate folder:**
   - `chronosx/` - ChronosX related
   - `modular_framework/` - Component tests
   - `models/` - Model-specific tests
   - `unit/` - Individual component tests

2. **Update test runner:**
   - Add to `run_all_tests.py` in `_define_test_suites()`
   - Specify category, priority, and estimated time

3. **Follow naming convention:**
   - `test_*.py` for test files
   - `demo_*.py` for demonstration scripts
   - `benchmark_*.py` for performance tests

## 🎯 Test Philosophy

Our testing strategy focuses on:

- ⚡ **Fast feedback** - Quick tests run first
- 🔮 **Real functionality** - Tests use actual models and data
- 🎯 **Practical scenarios** - Tests mirror real usage patterns
- 📊 **Performance awareness** - Track execution times and resources
- 🔧 **Easy maintenance** - Organized, documented, and modular

---

**Need help?** Run `python run_all_tests.py list` to see all available options!
