# Financial Data Manager - CSV/Parquet Support Documentation

## Overview

The `FinancialDataManager` in `data_provider/data_prepare.py` provides robust support for loading data files in both CSV and Parquet formats with intelligent fallback mechanisms.

## Features

### 1. **Multi-Format Support**
- **CSV files**: Standard comma-separated values
- **Parquet files**: Compressed, columnar format (requires `pyarrow`)
- **Auto-detection**: Automatically detects format when extension is unclear
- **Fallback mechanism**: If primary format fails, automatically tries the other format

### 2. **Format Detection Logic**

The `_load_file()` method implements the following logic:

```python
def _load_file(self, file_path: str) -> pd.DataFrame:
    """
    Load file supporting both CSV and Parquet formats.
    
    Detection hierarchy:
    1. Use file extension (.csv or .parquet)
    2. If unknown extension, try Parquet first, then CSV
    3. If primary format fails, try fallback format
    """
```

### 3. **Dependencies**

The system requires `pyarrow` for Parquet support:

```bash
pip install pyarrow
```

This is already included in `requirements.txt`.

### 4. **Usage Examples**

#### Basic Usage
```python
from data_provider.data_prepare import FinancialDataManager

# Initialize manager
data_manager = FinancialDataManager(data_root='data')

# Load CSV file
target_df = data_manager.load_target_data('nifty50_returns.csv')

# Load Parquet file
target_df = data_manager.load_target_data('nifty50_returns.parquet')

# Auto-detection (no extension)
target_df = data_manager.load_target_data('data_file')
```

#### Complete Pipeline
```python
# Prepare complete dataset with mixed formats
data_manager = FinancialDataManager(data_root='data')

prepared_data = data_manager.prepare_data(
    target_file='nifty50_returns.parquet',           # Parquet target
    dynamic_cov_file='dynamic_features.csv',         # CSV covariates
    static_cov_file='static_features.parquet',       # Parquet static
    alignment_method='forward_fill'
)
```

## 5. **Supported File Combinations**

| Target Data | Dynamic Covariates | Static Covariates | Status |
|-------------|-------------------|-------------------|---------|
| CSV         | CSV               | CSV               | ✅ Fully supported |
| Parquet     | Parquet           | Parquet           | ✅ Fully supported |
| CSV         | Parquet           | CSV               | ✅ Mixed formats supported |
| Parquet     | CSV               | Parquet           | ✅ Mixed formats supported |
| Any         | Any               | Any               | ✅ Auto-detection works |

## 6. **Error Handling**

The system provides comprehensive error handling:

- **File not found**: Clear error message with file path
- **Format detection failure**: Tries alternative formats
- **Corruption handling**: Graceful fallback to alternative format
- **Missing dependencies**: Clear error if `pyarrow` not installed

## 7. **Performance Considerations**

### Parquet Advantages:
- **Compression**: ~70% smaller file sizes
- **Speed**: Faster loading for large datasets
- **Type preservation**: Maintains data types exactly
- **Metadata**: Rich schema information

### CSV Advantages:
- **Human readable**: Easy to inspect and edit
- **Universal compatibility**: Works everywhere
- **No dependencies**: Standard pandas functionality
- **Debugging friendly**: Easy to validate data

## 8. **Test Results**

Our testing demonstrates:

1. **✅ Parquet Loading**: Successfully loads `.parquet` files
2. **✅ CSV Loading**: Successfully loads `.csv` files  
3. **✅ Auto-detection**: Handles files without clear extensions
4. **✅ Fallback Mechanism**: Gracefully handles format mismatches
5. **✅ Mixed Formats**: Supports different formats in same pipeline

### Test Output Summary:
```
Parquet file loaded successfully: (7109, 5)
CSV file loaded successfully: (7109, 6)
Auto-detection successful: (7109, 5)
Fallback mechanism worked: (7109, 5)
```

## 9. **Best Practices**

### For Production:
- Use **Parquet** for large datasets (>1GB)
- Use **CSV** for small datasets or human inspection
- Always specify file extensions when possible
- Test both formats in your pipeline

### For Development:
- Start with **CSV** for easy debugging
- Convert to **Parquet** for performance optimization
- Use mixed formats as needed for different data sources

## 10. **Integration with TSLib**

The data manager integrates seamlessly with the Time Series Library:

```python
# Standard TSLib workflow with either format
data_manager = FinancialDataManager(data_root='data')

# Load and prepare data (auto-detects formats)
prepared_data = data_manager.prepare_data(
    target_file='returns.parquet',     # or .csv
    dynamic_cov_file='features.csv',   # or .parquet  
    static_cov_file='static.parquet'   # or .csv
)

# Save in preferred format
data_manager.save_prepared_data('final_data.parquet', format='parquet')
```

This flexible format support ensures compatibility with various data sources and optimization opportunities while maintaining ease of use.
