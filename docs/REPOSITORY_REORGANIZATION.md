# Repository Reorganization Summary

## 📁 Directory Structure Changes

The repository has been reorganized for better maintainability and clarity:

### **Documentation** → `docs/`
All documentation files (except README.md and requirements.txt) moved to `docs/`:
- `AUTOFORMER_ENHANCEMENTS.md`
- `CONTRIBUTING.md` 
- `DECODER_BYPASS_ANALYSIS.md`
- `DIAGNOSTICS_GUIDE.md`
- `DYNAMIC_SYSTEM_COMPLETE.md`
- `ENHANCED_AUTOFORMER_MODES_GUIDE.md`
- `KL_TUNING_GUIDE.md`
- `MODEL_ANALYSIS_RECOMMENDATIONS.md`
- `SCALING_BUG_FIXES.md`
- `CSV_PARQUET_SUPPORT.md` (existing)

### **Training Scripts** → `scripts/train/`
All training scripts moved to `scripts/train/`:
- `train_bayesian_autoformer.py`
- `train_bayesian_with_kl_tuning.py`
- `train_configurable_autoformer.py`
- `train_dynamic_autoformer.py`
- `train_enhanced_autoformer.py`
- `train_financial_autoformer.py`
- `train_financial_timesnet.py`
- `train_financial_timesnet_v2.py`
- `train_hierarchical_autoformer.py`

### **Test Scripts** → `tests/`
All test and debug scripts moved to `tests/`:
- All `test_*.py` files
- All `debug_*.py` files
- All `*sanity*.py` files
- All verification scripts

### **Integration/Examples** → `integration/`
Examples, demos, utilities, and integration scripts moved to `integration/`:
- All `example_*.py` files
- All visualization files (`*.png`, notebooks)
- All demo and utility scripts
- Package creation scripts
- Configuration generators

### **Configuration Files** → `config/`
All configuration files moved to `config/`:
- All `config*.yaml` and `config*.json` files
- All `template_*.yaml` files

### **New Directory** → `trained/`
Created new directory for trained model artifacts:
- Model checkpoints
- Training logs
- Evaluation results
- Saved configurations

## 🔧 Path Updates

### **Training Scripts**
- Updated `sys.path` to point to root directory: `'..', '..'`
- Updated config file references to use `../../config/`
- Fixed imports to work from subdirectory

### **Test Scripts**
- Updated references to training scripts: `../scripts/train/`
- Updated references to config files: `../config/`

### **Integration Scripts**
- Updated references to training scripts: `../scripts/train/`
- Updated references to config files: `../config/`
- Updated package creation scripts with new paths

### **Documentation**
- Maintained README.md and requirements.txt in root
- All other documentation now organized in `docs/`

## ✅ Verification

- ✅ Training scripts can run from new location
- ✅ Config files accessible from all locations
- ✅ Import paths correctly updated
- ✅ Test scripts reference correct paths
- ✅ Integration scripts reference correct paths

## 🚀 Usage

### Running Training Scripts
```bash
# From root directory
python scripts/train/train_dynamic_autoformer.py --config config/config_enhanced_autoformer_MS_ultralight.yaml

# From scripts/train directory
cd scripts/train
python train_dynamic_autoformer.py --config ../../config/config_enhanced_autoformer_MS_ultralight.yaml
```

### Running Tests
```bash
# From root directory
python tests/test_all_enhanced_models.py

# From tests directory
cd tests
python test_all_enhanced_models.py
```

### Configuration Files
All configuration files are now centralized in `config/` directory and can be referenced using relative paths from any location.

## 📂 Final Structure

```
Time-Series-Library/
├── README.md                   # Main documentation
├── requirements.txt            # Dependencies
├── run.py                      # Main entry point
├── LICENSE                     # License
├── config/                     # All configuration files
├── docs/                       # All documentation
├── scripts/train/              # All training scripts
├── tests/                      # All test scripts
├── integration/                # Examples, demos, utilities
├── trained/                    # Trained model artifacts
├── data/                       # Data files
├── models/                     # Model implementations
├── layers/                     # Layer implementations
├── exp/                        # Experiment framework
├── utils/                      # Utility functions
├── data_provider/              # Data loading
└── [other core directories]
```

This reorganization provides:
- **Clear separation of concerns**
- **Centralized configuration management**
- **Organized documentation**
- **Structured training scripts**
- **Comprehensive testing framework**
- **Easy-to-find examples and demos**
