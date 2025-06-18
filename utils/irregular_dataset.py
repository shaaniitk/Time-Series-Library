"""
Enhanced Dataset Loader for Irregular Time Series

This module extends the existing data_loader.py to handle time series with:
- Different date ranges for different targets
- Missing data patterns
- Irregular sampling frequencies
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.temporal_alignment import TemporalDataAligner, IrregularDatasetLoader
from utils.logger import logger
from typing import List, Dict, Optional, Tuple, Union


class Dataset_Irregular(Dataset):
    """
    Dataset class that handles irregular time series data with different date ranges.
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='irregular_data.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None, fill_method='interpolate',
                 min_sequence_length=None):
        
        self.args = args
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.fill_method = fill_method
        self.min_sequence_length = min_sequence_length or self.seq_len

        self.root_path = root_path
        self.data_path = data_path
        
        logger.info(f"Initializing Dataset_Irregular for {flag} set")
        self.__read_data__()

    def __read_data__(self):
        """Read and process irregular time series data."""
        logger.info(f"Reading irregular data from {os.path.join(self.root_path, self.data_path)}")
        
        # Initialize components
        self.scaler = StandardScaler()
        aligner = TemporalDataAligner(freq=self.freq, fill_method=self.fill_method)
        
        # Load data
        try:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise

        # Handle target specification
        if isinstance(self.target, str):
            target_list = [t.strip() for t in self.target.split(',')]
        else:
            target_list = list(self.target) if hasattr(self.target, '__iter__') else [str(self.target)]
        
        # Process data based on structure
        if self._is_multi_file_scenario():
            # Multiple files scenario
            aligned_df = self._process_multi_file_data(aligner, target_list)
        else:
            # Single file with potential irregular patterns
            aligned_df = self._process_single_file_irregular_data(df_raw, aligner, target_list)
        
        # Create train/val/test splits accounting for irregular patterns
        self._create_irregular_splits(aligned_df, target_list)
        
        logger.info(f"Processed irregular data - Final shapes: data_x={self.data_x.shape}, data_y={self.data_y.shape}")

    def _is_multi_file_scenario(self):
        """Check if we're dealing with multiple files."""
        # This could be enhanced to detect multiple files in the directory
        return '*' in self.data_path or ',' in self.data_path

    def _process_multi_file_data(self, aligner, target_list):
        """Process multiple files with potentially different date ranges."""
        # Parse file paths
        if '*' in self.data_path:
            import glob
            file_paths = glob.glob(os.path.join(self.root_path, self.data_path))
        else:
            file_paths = [os.path.join(self.root_path, path.strip()) for path in self.data_path.split(',')]
        
        logger.info(f"Processing {len(file_paths)} files")
        
        # Load and align
        irregular_loader = IrregularDatasetLoader(aligner)
        aligned_df = irregular_loader.load_from_multiple_files(file_paths, target_list)
        
        return aligned_df

    def _process_single_file_irregular_data(self, df_raw, aligner, target_list):
        """Process single file with irregular patterns."""
        
        # Check for date column
        date_col = None
        for col in ['date', 'timestamp', 'time', 'Date', 'Timestamp', 'Time']:
            if col in df_raw.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("No date column found, using row index as date")
            df_raw['date'] = pd.date_range(start='2020-01-01', periods=len(df_raw), freq=self.freq)
            date_col = 'date'
        
        # Create data dictionary for alignment
        data_dict = {}
        
        # Process each target separately to handle different availability patterns
        for target in target_list:
            if target in df_raw.columns:
                # Extract non-null data for this target
                target_data = df_raw[[date_col, target]].dropna()
                if len(target_data) > 0:
                    data_dict[target] = target_data
                    logger.info(f"Target {target}: {len(target_data)} valid observations")
        
        # Add feature columns (covariates)
        feature_cols = [col for col in df_raw.columns 
                       if col not in target_list and col != date_col]
        if feature_cols:
            feature_data = df_raw[[date_col] + feature_cols]
            # Handle missing values in features differently (keep some missing patterns)
            data_dict['_features'] = feature_data
        
        # Align all data
        aligned_df = aligner.align_multiple_series(data_dict, date_col)
        
        return aligned_df

    def _create_irregular_splits(self, aligned_df, target_list):
        """Create train/val/test splits that account for data availability."""
        
        # Sort by date
        date_col = 'date'
        aligned_df = aligned_df.sort_values(date_col).reset_index(drop=True)
        
        # Calculate data availability for splitting
        target_cols = [col for col in aligned_df.columns if any(t in col for t in target_list)]
        
        # Find periods with sufficient data availability
        data_availability = aligned_df[target_cols].notna().sum(axis=1)
        min_targets_available = max(1, len(target_cols) // 2)  # At least half targets available
        
        valid_periods = data_availability >= min_targets_available
        valid_indices = aligned_df[valid_periods].index.tolist()
        
        logger.info(f"Found {len(valid_indices)} periods with sufficient data availability")
        
        # Create splits based on valid periods
        total_valid = len(valid_indices)
        if total_valid < self.seq_len + self.pred_len:
            raise ValueError(f"Insufficient valid data: {total_valid} < {self.seq_len + self.pred_len}")
        
        # Dynamic split ratios based on data availability
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        train_end = int(total_valid * train_ratio)
        val_end = int(total_valid * (train_ratio + val_ratio))
        
        if self.set_type == 0:  # train
            selected_indices = valid_indices[:train_end]
        elif self.set_type == 1:  # val
            selected_indices = valid_indices[train_end:val_end]
        else:  # test
            selected_indices = valid_indices[val_end:]
        
        # Extract data for selected indices
        selected_data = aligned_df.iloc[selected_indices].reset_index(drop=True)
        
        # Prepare features and targets based on mode
        if self.features == 'M' or self.features == 'MS':
            # Use all columns except date
            data_cols = [col for col in selected_data.columns if col != date_col]
            df_data = selected_data[data_cols]
        elif self.features == 'S':
            # Use only target columns
            df_data = selected_data[target_cols[:1]]  # First target only for S mode
        
        # Handle scaling
        if self.scale:
            if self.set_type == 0:  # fit scaler on training data
                self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Handle time encoding
        df_stamp = selected_data[[date_col]]
        df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[date_col].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp[date_col].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp[date_col].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp[date_col].apply(lambda row: row.hour)
            data_stamp = df_stamp.drop([date_col], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
        # Store validity mask for advanced sequence sampling
        self._create_validity_mask(selected_data, target_cols)

    def _create_validity_mask(self, data, target_cols):
        """Create a mask indicating valid sequences for sampling."""
        
        # Create mask for each possible sequence position
        data_availability = data[target_cols].notna().sum(axis=1)
        min_targets = max(1, len(target_cols) // 2)
        
        validity_mask = []
        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            # Check if sequence has sufficient data
            seq_slice = slice(i, i + self.seq_len + self.pred_len)
            seq_availability = data_availability.iloc[seq_slice]
            
            # Require minimum data availability throughout the sequence
            valid_ratio = (seq_availability >= min_targets).mean()
            validity_mask.append(valid_ratio >= 0.7)  # 70% of sequence must have data
        
        self.validity_mask = np.array(validity_mask)
        self.valid_indices = np.where(self.validity_mask)[0]
        
        logger.info(f"Valid sequences: {len(self.valid_indices)} out of {len(validity_mask)}")

    def __getitem__(self, index):
        """Get item with irregular data handling."""
        
        # Map to valid indices if using validity mask
        if hasattr(self, 'valid_indices') and len(self.valid_indices) > 0:
            actual_index = self.valid_indices[index % len(self.valid_indices)]
        else:
            actual_index = index
        
        s_begin = actual_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Handle any remaining NaN values
        seq_x = np.nan_to_num(seq_x, nan=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """Length accounting for irregular patterns."""
        if hasattr(self, 'valid_indices'):
            return len(self.valid_indices)
        else:
            return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        """Inverse transform with shape handling."""
        return self.scaler.inverse_transform(data)

    def get_data_info(self):
        """Get information about the loaded irregular data."""
        info = {
            'total_length': len(self.data_x),
            'valid_sequences': len(self.valid_indices) if hasattr(self, 'valid_indices') else 'N/A',
            'data_shape': self.data_x.shape,
            'time_shape': self.data_stamp.shape,
            'missing_ratio': np.isnan(self.data_x).mean() if self.data_x.dtype == float else 0.0
        }
        return info


# Integration function for existing framework
def create_irregular_data_loader(args, root_path, flag='train'):
    """
    Create a data loader that can handle irregular time series data.
    
    This function automatically detects whether to use regular or irregular loading
    based on the data characteristics.
    """
    
    # Try irregular loader first if data seems irregular
    try:
        dataset = Dataset_Irregular(
            args=args,
            root_path=root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            scale=args.scale,
            timeenc=args.timeenc,
            freq=args.freq
        )
        
        info = dataset.get_data_info()
        logger.info(f"Successfully loaded irregular dataset: {info}")
        
        return dataset
        
    except Exception as e:
        logger.warning(f"Irregular loader failed: {e}")
        logger.info("Falling back to regular dataset loader")
        
        # Fall back to regular loader
        from data_provider.data_loader import Dataset_Custom
        return Dataset_Custom(
            args=args,
            root_path=root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            scale=args.scale,
            timeenc=args.timeenc,
            freq=args.freq
        )
