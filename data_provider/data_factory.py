import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- Imports for the NEW robust pipeline ---
from data_provider.data_loader import ForecastingDataset
from data_provider.data_prepare import FinancialDataManager
from layers.modular.dimensions.dimension_manager import DimensionManager
from utils.scaler_manager import ScalerManager
from typing import Tuple # Ensure Tuple is imported
from utils.logger import logger

# --- Imports for the OLD pipeline (for backward compatibility) ---
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, 
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, 
    SWATSegLoader, UEAloader
)
from utils.timefeatures import time_features # Import time_features
from data_provider.uea import collate_fn

# --- NEW, ROBUST DATA PIPELINE FOR FINANCIAL FORECASTING ---
def setup_financial_forecasting_data(args):
    """
    The new, robust entry point for the specific financial forecasting task.
    It orchestrates loading, scaling, and dataloader creation using our new managers.
    """    # Step 1: Prepare the initial merged dataframe
    logger.info("Step 1: Preparing financial data using FinancialDataManager...") # Use FinancialDataManager to load and merge data
    fin_manager = FinancialDataManager(
        data_root=args.root_path,
        target_file=args.data_path, # Use args.data_path as the target file
        # dynamic_cov_file=args.dynamic_cov_path, # Assuming these are handled by prepare_data if not None
        # static_cov_file=args.static_cov_path
    )
    merged_df = fin_manager.prepare_data(
        target_file=args.data_path, # Pass again to ensure it's used
        # dynamic_cov_file=args.dynamic_cov_path,
        # static_cov_file=args.static_cov_path
    )

    # --- FIX: Enforce the target columns from the config ---
    # This ensures that the 'target' argument from the config file is the single source of truth,
    # overriding any columns inferred by the FinancialDataManager.
    fin_manager.target_columns = [col.strip() for col in args.target.split(',') if col.strip()]
    
    # All columns in the merged_df (including 'date')
    fin_manager.all_columns = merged_df.columns.tolist()
    logger.info(f"FinancialDataManager reports {len(fin_manager.target_columns)} target columns: {fin_manager.target_columns}")
    logger.info(f"FinancialDataManager reports {len(fin_manager.all_columns)} total columns (including date).")

    # Step 2: Create the DimensionManager
    logger.info("Step 2: Initializing DimensionManager...")
    # Filter out 'date' from all_features for dimension management
    all_features_no_date = [col for col in fin_manager.all_columns if col != 'date']
    dim_manager = DimensionManager(
        mode=args.features,
        target_features=fin_manager.target_columns,
        all_features=all_features_no_date, # Exclude date column from feature list for DM
        loss_function=args.loss,
        quantiles=getattr(args, 'quantile_levels', [])
    )
    logger.info(dim_manager)

    # Step 3: Split data into train/val/test sets
    logger.info("Step 3: Splitting data into train/val/test sets (chronological)...")
    df_train, df_val, df_test = _split_data(
        merged_df,
        args.validation_length,
        args.test_length,
        min_train_len=args.seq_len + args.pred_len,
        overlap_len=args.seq_len, # Provide history context for val/test
    )

    # Step 4: Create and fit the ScalerManager with the correct data scopes
    logger.info("Step 4: Fitting scalers with correct data scopes...")
    scaler_manager = ScalerManager(
        # Pass the actual lists of target and covariate features (excluding 'date')
        target_features=dim_manager.target_features,
        covariate_features=dim_manager.covariate_features
    )
    scaler_manager.fit(train_df=df_train, full_df=merged_df)
    
    # Step 5: Create datasets and dataloaders
    logger.info("Step 5: Creating datasets and dataloaders...")
    data_loaders = {}
    for flag, df_split in [('train', df_train), ('val', df_val), ('test', df_test)]:
        # ScalerManager.transform expects a DataFrame with rows; for empty splits, create empty arrays
        if df_split is None or len(df_split) == 0:
            logger.warning(f"Split '{flag}' is empty. Creating zero-length dataset.")
            # Determine expected feature counts
            if args.features == 'S':
                x_feat_count = len(dim_manager.target_features)
            else:
                # all features except date
                x_feat_count = len([c for c in merged_df.columns if c != 'date'])

            all_features_no_date = [c for c in merged_df.columns if c != 'date']
            y_feat_count = len(all_features_no_date)

            data_x_scaled = np.empty((0, x_feat_count), dtype=np.float32)
            data_y_unscaled = np.empty((0, y_feat_count), dtype=np.float32)
            data_stamp = np.empty((0, 0), dtype=np.float32)
        else:
            # For 'S' mode, data_x_scaled should only contain target features for the encoder input
            if args.features == 'S':
                data_x_scaled = scaler_manager.target_scaler.transform(df_split[dim_manager.target_features])
            else:
                # For 'M' and 'MS' modes, data_x_scaled includes all features
                data_x_scaled = scaler_manager.transform(df_split)

            # Get unscaled data as numpy array (excluding date)
            all_features_no_date = [col for col in df_split.columns if col != 'date']
            data_y_unscaled = df_split[all_features_no_date].values

            # Extract time features
            df_stamp = df_split[['date']].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date'])
            date_index = pd.DatetimeIndex(df_stamp['date'].values)
            logger.debug(f"Created DatetimeIndex with {len(date_index)} dates for time features")
            # Fixed time_features call with DatetimeIndex instead of Series
            data_stamp = time_features(date_index, freq=args.freq)
            data_stamp = data_stamp.transpose(1, 0) # Transpose to (num_dates, num_time_features)

        dataset = ForecastingDataset(data_x_scaled, data_y_unscaled, data_stamp, args, dim_manager)
        shuffle = True if flag == 'train' else False
        # Avoid dropping last to prevent empty batches on tiny datasets
        drop_last = False
        data_loaders[flag] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, drop_last=drop_last
        )
    logger.info("Data pipeline setup complete.")
    return data_loaders['train'], data_loaders['val'], data_loaders['test'], scaler_manager, dim_manager # Return the loaders and managers

def _split_data(
    df: pd.DataFrame,
    validation_length: int,
    test_length: int,
    *,
    min_train_len: int,
    overlap_len: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split into train/val/test honoring requested lengths with overlap.

    Rules:
    - Keep at least ``min_train_len`` rows for training (sliding-window minimum).
    - Try to honor ``validation_length`` and ``test_length`` exactly where possible.
    - If defined, ``overlap_len`` (usually seq_len) is PREPENDED to val and test sets
      to provide history context. This allows val/test sets to generate valid windows
      starting from their first real data point.
    """
    total_len = len(df)

    # Start with requested lengths (non-negative)
    num_val_target = max(0, int(validation_length))
    num_test_target = max(0, int(test_length))

    # Ensure at least the required training rows remain.
    # Note: min_train_len is basically just seq_len + pred_len usually.
    required = max(1, int(min_train_len))
    
    # We first calculate the "core" non-overlapping sizes
    # Reduce val first, then test, until the constraint is satisfied
    current_val = num_val_target
    current_test = num_test_target
    
    while total_len - (current_val + current_test) < required and (current_val > 0 or current_test > 0):
        if current_val > 0:
            current_val -= 1
        elif current_test > 0:
            current_test -= 1

    # whatever remains is training
    num_train = max(required, total_len - current_val - current_test)
    
    # Materialize splits with OVERLAP
    # Train: [0 : num_train]
    df_train = df.iloc[0:num_train].copy()
    
    # Val: [num_train - overlap : num_train + current_val]
    # We start 'overlap_len' before the validation block starts
    val_start = max(0, num_train - overlap_len)
    val_end = min(total_len, num_train + current_val)
    df_val = df.iloc[val_start:val_end].copy()
    
    # Test: [num_train + current_val - overlap : num_train + current_val + current_test]
    test_start = max(0, num_train + current_val - overlap_len)
    test_end = min(total_len, num_train + current_val + current_test)
    df_test = df.iloc[test_start:test_end].copy()
    
    return df_train, df_val, df_test
# --- END OF NEW PIPELINE ---

# --- LEGACY PIPELINE FOR BACKWARD COMPATIBILITY ---
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}

def data_provider(args, flag, dataset_obj=None):
    """
    Legacy entry point for non-financial or legacy tasks.
    Can now accept a pre-initialized dataset object to avoid reloading data.
    """
    if dataset_obj is not None:
        data_set = dataset_obj
    else:
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1
        
        if args.task_name == 'anomaly_detection':
            data_set = Data(
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
            )
        elif args.task_name == 'classification':
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
            )
        else:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=args.freq,
                seasonal_patterns=getattr(args, 'seasonal_patterns', 'Monthly')
            )
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.task_name == 'classification':
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
            
    return data_set, data_loader