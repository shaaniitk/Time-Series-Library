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
    df_train, df_val, df_test = _split_data(merged_df, args.validation_length, args.test_length)

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
        # ScalerManager.transform expects a DataFrame, it will select columns internally
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
        drop_last = True if flag == 'train' else False # Only drop last for training
        data_loaders[flag] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, drop_last=drop_last
        )
    logger.info("Data pipeline setup complete.")
    return data_loaders['train'], data_loaders['val'], data_loaders['test'], scaler_manager, dim_manager # Return the loaders and managers

def _split_data(df: pd.DataFrame, validation_length: int, test_length: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper to split a dataframe chronologically into train, validation, and test sets.
    The validation and test sets are taken from the end of the dataframe.
    """
    total_len = len(df)
    num_test = test_length
    num_val = validation_length
    num_train = total_len - num_val - num_test
    df_train = df.iloc[0:num_train].copy()
    df_val = df.iloc[num_train:num_train + num_val].copy()
    df_test = df.iloc[num_train + num_val:].copy()
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

def data_provider(args, flag):
    """
    Legacy entry point for non-financial or legacy tasks.
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
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
            freq=freq,
            seasonal_patterns=getattr(args, 'seasonal_patterns', 'Monthly')
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader