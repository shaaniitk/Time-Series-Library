"""
Custom Data Preparation Manager for Financial Time Series
Handles target data (business days), dynamic covariates (daily), and static covariates.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import os
from utils.logger import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.normalization import TSNormalizer


class FinancialDataManager:
    """Custom data manager for financial time series.

    Responsibilities:
      * Load target, dynamic, and static covariate datasets.
      * Align datasets on a common business-day date axis.
      * Optionally apply global normalization to numeric columns (including
        targets and covariates) via a unified TSNormalizer abstraction.
    """

    def __init__(
        self,
        data_root: str = 'data',
        target_file: Optional[str] = None,
        dynamic_cov_file: Optional[str] = None,
        static_cov_file: Optional[str] = None,
        enable_normalization: bool = False,
        normalization_mode: str = 'standard',
    ) -> None:
        """Initialize the manager.

        Parameters
        ----------
        data_root : str
            Root directory containing data assets.
        target_file : Optional[str]
            Primary target series file.
        dynamic_cov_file : Optional[str]
            High-frequency / daily covariates file.
        static_cov_file : Optional[str]
            Time-invariant covariates file.
        enable_normalization : bool
            If True, apply normalization after alignment.
        normalization_mode : str
            Normalization strategy (see TSNormalizer.SUPPORTED).
        """
        self.data_root = data_root
        self.target_file = target_file
        self.dynamic_cov_file = dynamic_cov_file
        self.static_cov_file = static_cov_file

        # Normalization configuration
        self.enable_normalization = enable_normalization
        self.normalization_mode = normalization_mode
        self.normalizer: Optional[TSNormalizer] = None

        # Data containers
        self.target_data: Optional[pd.DataFrame] = None
        self.dynamic_cov_data: Optional[pd.DataFrame] = None
        self.static_cov_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.merged_data_normalized: Optional[pd.DataFrame] = None

        # Metadata
        self.target_columns: list[str] = []
        self.dynamic_cov_columns: list[str] = []
        self.static_cov_columns: list[str] = []
        self.date_column: str = 'date'

        logger.info(f"Initialized FinancialDataManager with data_root: {data_root}")
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load file supporting both CSV and Parquet formats.
        Handles date columns and date indexes properly.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.parquet':
                logger.info(f"Loading Parquet file: {file_path}")
                df = pd.read_parquet(file_path)
            elif file_ext == '.csv':
                logger.info(f"Loading CSV file: {file_path}")
                df = pd.read_csv(file_path)
            else:
                # Try to infer format from content
                logger.warning(f"Unknown file extension {file_ext}, trying to auto-detect format")
                try:
                    # Try parquet first
                    df = pd.read_parquet(file_path)
                    logger.info("Successfully loaded as Parquet")
                except:
                    # Fall back to CSV
                    df = pd.read_csv(file_path)
                    logger.info("Successfully loaded as CSV")
                    
        except Exception as e:
            # If primary format fails, try the other format
            logger.warning(f"Failed to load {file_path} as {file_ext}: {e}")
            
            if file_ext == '.parquet':
                logger.info("Trying to load as CSV instead...")
                df = pd.read_csv(file_path)
            elif file_ext == '.csv':
                logger.info("Trying to load as Parquet instead...")
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Could not load file {file_path} in any supported format")
        
        # Handle date index vs date column
        df = self._handle_date_index(df)
                
        logger.info(f"Successfully loaded file: {df.shape}")
        return df
    
    def _handle_date_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle files where date might be in index or column.
        Converts date index to a 'date' column for consistent handling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with date as a column
        """
        # Check if index is a datetime type
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info("Found datetime index, converting to 'date' column")
            df = df.reset_index()
            # Find the index column (usually 'index' or similar)
            index_cols = [col for col in df.columns if col.lower() in ['index', 'date', 'time', 'datetime']]
            if index_cols:
                index_col = index_cols[0]
                df = df.rename(columns={index_col: 'date'})
                logger.info(f"Renamed index column '{index_col}' to 'date'")
            else:
                # If no obvious index column, assume first column is the date
                first_col = df.columns[0]
                df = df.rename(columns={first_col: 'date'})
                logger.info(f"Assumed first column '{first_col}' is date, renamed to 'date'")
        
        # Check if index looks like a date (string that can be parsed)
        elif df.index.dtype == 'object':
            try:
                # Try to parse index as dates
                pd.to_datetime(df.index[:5])  # Test first 5 values
                logger.info("Found string-based date index, converting to 'date' column")
                df = df.reset_index()
                index_col = df.columns[0]
                df = df.rename(columns={index_col: 'date'})
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Converted index to 'date' column")
            except (ValueError, TypeError):                # Index is not date-like, leave as is
                pass
        
        return df
    
    def load_target_data(self, target_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load target data (business days frequency).
        
        Args:
            target_file: Path to target file, if None uses self.target_file
            
        Returns:
            DataFrame with target data
        """
        if target_file is None:
            target_file = self.target_file
            
        if target_file is None:
            raise ValueError("No target file specified")
            
        file_path = os.path.join(self.data_root, target_file)
        logger.info(f"Loading target data from: {file_path}")
        
        df = self._load_file(file_path)
        
        # Ensure date column exists and is properly formatted
        # _load_file should have already handled date index -> column conversion
        if 'date' in df.columns:
            # Be flexible: handle ISO8601, mixed formats, or day-first with inference
            try:
                df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='raise')
            except Exception:
                try:
                    df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='raise')
                except Exception:
                    # Fallback to mixed inference, prefer dayfirst if ambiguous
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce', infer_datetime_format=False)
                    if df['date'].isna().any():
                        # As a last resort, try non-dayfirst
                        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
                    if df['date'].isna().any():
                        raise ValueError("Failed to parse some date values; please ensure date column is parseable.")
            df = df.sort_values('date').reset_index(drop=True)
            self.date_column = 'date'
        else:
            # Try to find date column (case insensitive) as fallback
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                original_date_col = date_cols[0]
                # Rename to standard 'date' column
                df = df.rename(columns={original_date_col: 'date'})
                self.date_column = 'date'
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df = df.sort_values(self.date_column).reset_index(drop=True)
                logger.info(f"Using date column: {original_date_col} -> {self.date_column}")
            else:
                raise ValueError("No date column found in target data after processing")
        
        # Identify target columns (numeric columns excluding date)
        self.target_columns = [col for col in df.columns 
                              if col != self.date_column and pd.api.types.is_numeric_dtype(df[col])]
        
        logger.info(f"Loaded target data: {df.shape}, Target columns: {self.target_columns}")
        logger.info(f"Date range: {df[self.date_column].min()} to {df[self.date_column].max()}")        
        self.target_data = df
        return df
    
    def load_dynamic_covariates(self, dynamic_cov_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load dynamic covariates data (daily frequency).
        
        Args:
            dynamic_cov_file: Path to dynamic covariates file
            
        Returns:
            DataFrame with dynamic covariates
        """
        if dynamic_cov_file is None:
            dynamic_cov_file = self.dynamic_cov_file
            
        if dynamic_cov_file is None:
            logger.info("No dynamic covariates file specified, skipping")
            return pd.DataFrame()
            
        file_path = os.path.join(self.data_root, dynamic_cov_file)
        logger.info(f"Loading dynamic covariates from: {file_path}")
        
        df = self._load_file(file_path)
        
        # Ensure date column exists and is properly formatted
        # _load_file should have already handled date index -> column conversion
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            self.date_column = 'date'
        else:
            # Try to find date column (case insensitive) as fallback
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                original_date_col = date_cols[0]
                # Rename to standard 'date' column
                df = df.rename(columns={original_date_col: 'date'})
                self.date_column = 'date'
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df = df.sort_values(self.date_column).reset_index(drop=True)
                logger.info(f"Using date column: {original_date_col} -> {self.date_column}")
            else:
                logger.warning("No date column found in dynamic covariates")
          # Identify dynamic covariate columns and add prefix to avoid conflicts
        numeric_cols = [col for col in df.columns 
                       if col != self.date_column and pd.api.types.is_numeric_dtype(df[col])]
        
        # Add 'dyn_' prefix to dynamic covariate columns
        rename_dict = {col: f'dyn_{col}' for col in numeric_cols}
        df = df.rename(columns=rename_dict)
        self.dynamic_cov_columns = [f'dyn_{col}' for col in numeric_cols]
        
        logger.info(f"Loaded dynamic covariates: {df.shape}, Columns: {len(self.dynamic_cov_columns)}")
        if 'date' in df.columns:
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        self.dynamic_cov_data = df
        return df
    
    def load_static_covariates(self, static_cov_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load static covariates data (time-invariant features).
        
        Args:
            static_cov_file: Path to static covariates file
            
        Returns:
            DataFrame with static covariates
        """
        if static_cov_file is None:
            static_cov_file = self.static_cov_file
            
        if static_cov_file is None:
            logger.info("No static covariates file specified, skipping")
            return pd.DataFrame()
            
        file_path = os.path.join(self.data_root, static_cov_file)
        logger.info(f"Loading static covariates from: {file_path}")
        
        df = self._load_file(file_path)
        
        # Process static covariates
        # These are time-invariant, so we'll need to broadcast them to match target dates
        self.static_cov_columns = [col for col in df.columns 
                                  if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
            for col in categorical_cols:
                if col in self.static_cov_columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        logger.info(f"Loaded static covariates: {df.shape}, Columns: {self.static_cov_columns}")
        
        self.static_cov_data = df
        return df
    
    def align_data_by_dates(self, 
                           method: str = 'forward_fill',
                           max_gap_days: int = 7) -> pd.DataFrame:
        """
        Align dynamic covariates (daily) with target data dates (business days).
        
        Args:
            method: Alignment method ('forward_fill', 'backward_fill', 'interpolate', 'nearest')
            max_gap_days: Maximum allowed gap in days for filling
            
        Returns:
            DataFrame with aligned data
        """
        if self.target_data is None:
            raise ValueError("Target data must be loaded first")
            
        logger.info(f"Aligning data using method: {method}")
        
        # Start with target data dates
        target_dates = self.target_data[self.date_column].copy()
        aligned_data = self.target_data.copy()
        
        # Align dynamic covariates if available
        if self.dynamic_cov_data is not None and not self.dynamic_cov_data.empty:
            logger.info("Aligning dynamic covariates with target dates")
            
            # Create a complete date range from min to max
            min_date = min(target_dates.min(), self.dynamic_cov_data[self.date_column].min())
            max_date = max(target_dates.max(), self.dynamic_cov_data[self.date_column].max())
            
            # Merge dynamic covariates with target dates
            dynamic_aligned = []
            
            for target_date in target_dates:
                # Find the best matching covariate data for this target date
                if method == 'forward_fill':
                    # Use the most recent covariate data before or on target_date
                    mask = self.dynamic_cov_data[self.date_column] <= target_date
                    if mask.any():
                        cov_row = self.dynamic_cov_data[mask].iloc[-1]
                        gap_days = (target_date - cov_row[self.date_column]).days
                        if gap_days <= max_gap_days:
                            dynamic_aligned.append(cov_row[self.dynamic_cov_columns].values)
                        else:
                            dynamic_aligned.append([np.nan] * len(self.dynamic_cov_columns))
                    else:
                        dynamic_aligned.append([np.nan] * len(self.dynamic_cov_columns))
                        
                elif method == 'backward_fill':
                    # Use the next available covariate data on or after target_date
                    mask = self.dynamic_cov_data[self.date_column] >= target_date
                    if mask.any():
                        cov_row = self.dynamic_cov_data[mask].iloc[0]
                        gap_days = (cov_row[self.date_column] - target_date).days
                        if gap_days <= max_gap_days:
                            dynamic_aligned.append(cov_row[self.dynamic_cov_columns].values)
                        else:
                            dynamic_aligned.append([np.nan] * len(self.dynamic_cov_columns))
                    else:
                        dynamic_aligned.append([np.nan] * len(self.dynamic_cov_columns))
                        
                elif method == 'nearest':
                    # Use the nearest covariate data to target_date
                    time_diffs = abs(self.dynamic_cov_data[self.date_column] - target_date)
                    nearest_idx = time_diffs.idxmin()
                    gap_days = time_diffs.min().days
                    if gap_days <= max_gap_days:
                        cov_row = self.dynamic_cov_data.iloc[nearest_idx]
                        dynamic_aligned.append(cov_row[self.dynamic_cov_columns].values)
                    else:
                        dynamic_aligned.append([np.nan] * len(self.dynamic_cov_columns))
            
            # Add dynamic covariates to aligned data
            dynamic_df = pd.DataFrame(dynamic_aligned, columns=self.dynamic_cov_columns)
            aligned_data = pd.concat([aligned_data, dynamic_df], axis=1)
            
            logger.info(f"Added {len(self.dynamic_cov_columns)} dynamic covariate columns")
        
        # Add static covariates if available
        if self.static_cov_data is not None and not self.static_cov_data.empty:
            logger.info("Adding static covariates (broadcasted to all dates)")
            
            # Broadcast static covariates to all target dates
            # Assuming we take the first row of static data (or you can specify logic)
            static_row = self.static_cov_data.iloc[0]
            for col in self.static_cov_columns:
                aligned_data[col] = static_row[col]
                
            logger.info(f"Added {len(self.static_cov_columns)} static covariate columns")
        
        # Handle missing values
        initial_na_count = aligned_data.isna().sum().sum()
        if initial_na_count > 0:
            logger.warning(f"Found {initial_na_count} missing values after alignment")
            
            # Forward fill, then backward fill to handle remaining NAs
            aligned_data = aligned_data.fillna(method='ffill').fillna(method='bfill')
            
            final_na_count = aligned_data.isna().sum().sum()
            if final_na_count > 0:
                logger.warning(f"Still have {final_na_count} missing values after filling")
                # Fill remaining with column means for numeric columns
                numeric_cols = aligned_data.select_dtypes(include=[np.number]).columns
                aligned_data[numeric_cols] = aligned_data[numeric_cols].fillna(aligned_data[numeric_cols].mean())
        
        logger.info(f"Final aligned data shape: {aligned_data.shape}")
        logger.info(f"Columns: Target({len(self.target_columns)}), Dynamic({len(self.dynamic_cov_columns)}), Static({len(self.static_cov_columns)})")
        
        self.merged_data = aligned_data

        # Optional normalization step (post alignment, includes targets & covariates)
        if self.enable_normalization:
            try:
                self.normalizer = TSNormalizer(self.normalization_mode)
                self.normalizer.fit(self.merged_data)
                self.merged_data_normalized = self.normalizer.transform(self.merged_data)
                logger.info(f"Applied normalization mode='{self.normalization_mode}' to merged data.")
            except Exception as e:
                logger.warning(f"Normalization skipped due to error: {e}")
                self.merged_data_normalized = None

        return aligned_data
    
    def prepare_data(self, 
                    target_file: Optional[str] = None,
                    dynamic_cov_file: Optional[str] = None,
                    static_cov_file: Optional[str] = None,
                    alignment_method: str = 'forward_fill',
                    max_gap_days: int = 7) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            target_file: Target data file path
            dynamic_cov_file: Dynamic covariates file path  
            static_cov_file: Static covariates file path
            alignment_method: Method for aligning dynamic covariates with target dates
            max_gap_days: Maximum allowed gap for alignment
            
        Returns:
            Complete prepared dataset
        """
        logger.info("Starting complete data preparation pipeline")
        
        # Load all data
        self.load_target_data(target_file)
        self.load_dynamic_covariates(dynamic_cov_file)
        self.load_static_covariates(static_cov_file)
        
        # Align data by dates
        merged_data = self.align_data_by_dates(alignment_method, max_gap_days)
        
        logger.info("Data preparation completed successfully")
        return merged_data
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the prepared data."""
        if self.merged_data is None:
            return {"error": "No data prepared yet"}
            
        info = {
            "total_shape": self.merged_data.shape,
            "date_range": {
                "start": self.merged_data[self.date_column].min(),
                "end": self.merged_data[self.date_column].max(),
                "total_days": (self.merged_data[self.date_column].max() - 
                              self.merged_data[self.date_column].min()).days
            },
            "columns": {
                "target": self.target_columns,
                "dynamic_covariates": self.dynamic_cov_columns,
                "static_covariates": self.static_cov_columns
            },
            "missing_values": self.merged_data.isna().sum().to_dict(),
            "data_types": self.merged_data.dtypes.to_dict(),
            "normalization": {
                "enabled": self.enable_normalization,
                "mode": self.normalization_mode,
                "fitted": self.normalizer.fitted() if self.normalizer else False,
                "stats": self.normalizer.info() if self.normalizer and self.normalizer.fitted() else {}
            }
        }
        return info
    
    def save_prepared_data(self, output_path: str, format: str = 'csv'):
        """
        Save the prepared data to file.
        
        Args:
            output_path: Output file path
            format: Output format ('csv' or 'parquet')
        """
        if self.merged_data is None:
            raise ValueError("No data prepared yet")
            
        if format.lower() == 'csv':
            self.merged_data.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            self.merged_data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Saved prepared data to: {output_path}")
