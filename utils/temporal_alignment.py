"""
Multi-Target Temporal Alignment Utilities

This module provides utilities for handling time series with different date ranges,
missing data, and irregular sampling patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from utils.logger import logger


class TemporalDataAligner:
    """
    Aligns multiple time series with different date ranges and missing data.
    """
    
    def __init__(self, freq='D', fill_method='interpolate'):
        """
        Args:
            freq: Target frequency ('D', 'H', 'W', 'M', etc.)
            fill_method: How to handle missing values ('interpolate', 'forward', 'backward', 'zero', 'mean')
        """
        self.freq = freq
        self.fill_method = fill_method
        logger.info(f"Initializing TemporalDataAligner with freq={freq}, fill_method={fill_method}")
    
    def align_multiple_series(self, data_dict: Dict[str, pd.DataFrame], 
                            date_col: str = 'date',
                            common_start: Optional[str] = None,
                            common_end: Optional[str] = None) -> pd.DataFrame:
        """
        Align multiple time series to a common date range.
        
        Args:
            data_dict: Dictionary of {target_name: dataframe}
            date_col: Name of the date column
            common_start: Start date for alignment (if None, uses earliest available)
            common_end: End date for alignment (if None, uses latest available)
            
        Returns:
            Aligned DataFrame with all targets
        """
        logger.info(f"Aligning {len(data_dict)} time series")
        
        # Determine common date range
        if common_start is None or common_end is None:
            all_dates = []
            for name, df in data_dict.items():
                if date_col in df.columns:
                    all_dates.extend(pd.to_datetime(df[date_col]).tolist())
            
            if common_start is None:
                common_start = min(all_dates)
            if common_end is None:
                common_end = max(all_dates)
        
        # Create common date index
        common_dates = pd.date_range(start=common_start, end=common_end, freq=self.freq)
        logger.info(f"Common date range: {common_start} to {common_end} ({len(common_dates)} periods)")
        
        # Align each series
        aligned_dfs = []
        for target_name, df in data_dict.items():
            aligned_df = self._align_single_series(df, target_name, common_dates, date_col)
            aligned_dfs.append(aligned_df)
        
        # Combine all series
        result = aligned_dfs[0]
        for df in aligned_dfs[1:]:
            result = result.merge(df, on=date_col, how='outer')
        
        # Sort by date and reset index
        result = result.sort_values(date_col).reset_index(drop=True)
        
        logger.info(f"Aligned data shape: {result.shape}")
        return result
    
    def _align_single_series(self, df: pd.DataFrame, target_name: str, 
                           common_dates: pd.DatetimeIndex, date_col: str) -> pd.DataFrame:
        """Align a single time series to the common date range."""
        
        # Prepare the series
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Create target dataframe with common dates
        target_df = pd.DataFrame({date_col: common_dates})
        
        # Merge with existing data
        merged = target_df.merge(df_copy, on=date_col, how='left')
        
        # Handle missing values for all non-date columns
        value_cols = [col for col in merged.columns if col != date_col]
        
        for col in value_cols:
            merged[col] = self._fill_missing_values(merged[col])
            
            # Rename column to include target name if not already named
            if not col.startswith(target_name):
                new_col_name = f"{target_name}_{col}" if col != target_name else col
                merged = merged.rename(columns={col: new_col_name})
        
        return merged
    
    def _fill_missing_values(self, series: pd.Series) -> pd.Series:
        """Fill missing values according to the specified method."""
        
        if self.fill_method == 'interpolate':
            return series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        elif self.fill_method == 'forward':
            return series.fillna(method='ffill')
        elif self.fill_method == 'backward':
            return series.fillna(method='bfill')
        elif self.fill_method == 'zero':
            return series.fillna(0)
        elif self.fill_method == 'mean':
            return series.fillna(series.mean())
        else:
            raise ValueError(f"Unknown fill_method: {self.fill_method}")


class IrregularDatasetLoader:
    """
    Dataset loader that handles irregular time series data.
    """
    
    def __init__(self, aligner: TemporalDataAligner):
        self.aligner = aligner
    
    def load_from_multiple_files(self, file_paths: List[str], 
                               target_names: Optional[List[str]] = None,
                               date_col: str = 'date') -> pd.DataFrame:
        """
        Load and align data from multiple CSV files.
        
        Args:
            file_paths: List of paths to CSV files
            target_names: Optional list of target names (defaults to filenames)
            date_col: Name of the date column
            
        Returns:
            Aligned DataFrame
        """
        logger.info(f"Loading {len(file_paths)} files")
        
        if target_names is None:
            target_names = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
        
        data_dict = {}
        for i, (path, name) in enumerate(zip(file_paths, target_names)):
            try:
                df = pd.read_csv(path)
                data_dict[name] = df
                logger.debug(f"Loaded {name}: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue
        
        return self.aligner.align_multiple_series(data_dict, date_col)
    
    def load_from_single_file_multiple_targets(self, file_path: str,
                                             target_cols: List[str],
                                             date_col: str = 'date') -> pd.DataFrame:
        """
        Load data from a single file but handle different availability for each target.
        
        Args:
            file_path: Path to the CSV file
            target_cols: List of target column names
            date_col: Name of the date column
            
        Returns:
            Processed DataFrame ready for time series modeling
        """
        logger.info(f"Loading multi-target data from {file_path}")
        
        df = pd.read_csv(file_path)
        df[date_col] = pd.to_datetime(df[date_col])
        
        # For each target, create a separate series and then align
        data_dict = {}
        for target in target_cols:
            if target in df.columns:
                # Create a series with only non-null values for this target
                target_data = df[[date_col, target]].dropna()
                data_dict[target] = target_data
                logger.debug(f"Target {target}: {len(target_data)} valid observations")
        
        # Add any additional feature columns (covariates)
        feature_cols = [col for col in df.columns if col not in target_cols and col != date_col]
        if feature_cols:
            # Include features for alignment
            feature_data = df[[date_col] + feature_cols]
            data_dict['_features'] = feature_data
        
        aligned = self.aligner.align_multiple_series(data_dict, date_col)
        
        # Separate targets and features in the final output
        return aligned


# Usage examples and utilities
def create_sample_irregular_data():
    """Create sample data with different date ranges for testing."""
    
    # Target A: 2020-2023 (full range)
    dates_a = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    target_a = pd.DataFrame({
        'date': dates_a,
        'target_a': np.random.randn(len(dates_a)) * 10 + 100
    })
    
    # Target B: 2021-2024 (different range, some missing)
    dates_b = pd.date_range('2021-06-01', '2024-06-30', freq='D')
    target_b = pd.DataFrame({
        'date': dates_b,
        'target_b': np.random.randn(len(dates_b)) * 15 + 200
    })
    # Introduce some random missing values
    missing_mask = np.random.random(len(target_b)) < 0.1
    target_b.loc[missing_mask, 'target_b'] = np.nan
    
    # Target C: 2019-2022 (earlier start, earlier end)
    dates_c = pd.date_range('2019-01-01', '2022-12-31', freq='D')
    target_c = pd.DataFrame({
        'date': dates_c,
        'target_c': np.random.randn(len(dates_c)) * 8 + 50
    })
    
    return {'target_a': target_a, 'target_b': target_b, 'target_c': target_c}


if __name__ == "__main__":
    # Example usage
    logger.info("Testing TemporalDataAligner")
    
    # Create sample irregular data
    sample_data = create_sample_irregular_data()
    
    # Initialize aligner
    aligner = TemporalDataAligner(freq='D', fill_method='interpolate')
    
    # Align the data
    aligned_data = aligner.align_multiple_series(sample_data)
    
    logger.info(f"Aligned data shape: {aligned_data.shape}")
    logger.info(f"Date range: {aligned_data['date'].min()} to {aligned_data['date'].max()}")
    logger.info(f"Columns: {list(aligned_data.columns)}")
    
    # Check for missing values
    missing_counts = aligned_data.isnull().sum()
    logger.info(f"Missing values after alignment: {dict(missing_counts)}")
    
    print("\nFirst 10 rows:")
    print(aligned_data.head(10))
    print("\nLast 10 rows:")
    print(aligned_data.tail(10))
