import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Optional, Tuple

# ... (imports and class definition remain the same) ...

class ScalerManager:
    def __init__(self, target_features: List[str], covariate_features: List[str], scaler_type: str = 'standard'):
        self.target_features = target_features
        # Filter out 'date' from covariate features to avoid scaling issues
        self.covariate_features = [col for col in covariate_features if col != 'date']
        self.scaler_type = scaler_type.lower()
        
        # Print debug info
        print(f"ScalerManager initialized with {len(self.target_features)} target features and {len(self.covariate_features)} covariate features. Type: {self.scaler_type}")
        
        if self.scaler_type == 'robust':
            self.target_scaler = RobustScaler()
            self.scaler: Optional[object] = RobustScaler() if self.covariate_features else None
        else:
            self.target_scaler = StandardScaler()
            self.scaler: Optional[object] = StandardScaler() if self.covariate_features else None
            
        self._target_stats_cache: Dict[Tuple[torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cov_stats_cache: Dict[Tuple[torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(self, train_df: pd.DataFrame, full_df: pd.DataFrame):
        """
        Fits the scalers according to the specified data-scoping rules.
        - Target scaler is fit ONLY on the training data to prevent leakage.
        - Covariate scaler is fit on the FULL dataset for global normalization.
        """
        print("Fitting scalers...")
        print(f"  - Fitting target scaler on {len(train_df)} training rows.")
        self.target_scaler.fit(train_df[self.target_features])
        
        if self.scaler:
            print(f"  - Fitting covariate scaler on {len(full_df)} total rows.")
            self.scaler.fit(full_df[self.covariate_features])
        
        print("Scalers fitted successfully.")
        self._target_stats_cache.clear()
        self._cov_stats_cache.clear()
    
    # ... (the transform and inverse_transform_targets methods remain exactly the same) ...
# ... existing code ...

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Scales a dataframe and returns a combined numpy array [targets, covariates]."""
        scaled_targets = self.target_scaler.transform(df[self.target_features])
        if self.scaler:
            scaled_covariates = self.scaler.transform(df[self.covariate_features])
            return np.concatenate([scaled_targets, scaled_covariates], axis=1)
        return scaled_targets

    def inverse_transform_targets(self, data: np.ndarray) -> np.ndarray: # This method is correct for targets
        """
        Robustly inverse-transforms target data, handling quantile dimensions.
        This is the method that prevents evaluation errors.
        """
        # Check if the input data has quantile predictions
        # e.g., data shape is [batch, 4 targets * 9 quantiles]
        if data.shape[-1] != len(self.target_features):
            num_quantiles = data.shape[-1] // len(self.target_features)
            if data.shape[-1] % len(self.target_features) == 0 and num_quantiles > 1:
                # Reshape from [batch, targets*quantiles] to [batch*quantiles, targets]
                original_shape = data.shape
                data_reshaped = data.reshape(-1, len(self.target_features))
                unscaled_reshaped = self.target_scaler.inverse_transform(data_reshaped)
                # Reshape back to the original [batch, targets*quantiles]
                return unscaled_reshaped.reshape(original_shape)
        
        # Standard case for point predictions
        return self.target_scaler.inverse_transform(data)

    def inverse_transform_all_features(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse-transforms data containing both target and covariate features,
        using their respective scalers.
        Assumes data columns are ordered as [targets, covariates].
        """
        num_targets = len(self.target_features)
        
        # Split the data into target and covariate parts
        target_data_scaled = data[:, :num_targets]
        covariate_data_scaled = data[:, num_targets:]
        
        # Inverse transform the target part
        target_data_unscaled = self.inverse_transform_targets(target_data_scaled)
        
        # Inverse transform the covariate part if it exists
        if self.scaler and covariate_data_scaled.shape[1] > 0:
            covariate_data_unscaled = self.scaler.inverse_transform(covariate_data_scaled)
            # Combine the unscaled parts
            return np.concatenate([target_data_unscaled, covariate_data_unscaled], axis=1)
        else:
            # If no covariates, just return the unscaled targets
            return target_data_unscaled

    # ------------------------------------------------------------------
    # Torch-native scaling helpers to avoid repeated numpy round-trips
    # ------------------------------------------------------------------

    def _get_target_stats_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._target_stats_cache.get(key)
        if cached is None:
            if hasattr(self.target_scaler, 'center_'): # RobustScaler
                 mean_val = self.target_scaler.center_
                 scale_val = self.target_scaler.scale_
            else: # StandardScaler
                 mean_val = self.target_scaler.mean_
                 scale_val = self.target_scaler.scale_

            mean = torch.as_tensor(mean_val, device=device, dtype=dtype)
            scale = torch.as_tensor(scale_val, device=device, dtype=dtype)
            self._target_stats_cache[key] = (mean, scale)
            return mean, scale
        return cached

    def _get_cov_stats_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None:
            raise RuntimeError("Covariate scaler is not initialised.")
        key = (device, dtype)
        cached = self._cov_stats_cache.get(key)
        if cached is None:
            if hasattr(self.scaler, 'center_'): # RobustScaler
                 mean_val = self.scaler.center_
                 scale_val = self.scaler.scale_
            else: # StandardScaler
                 mean_val = self.scaler.mean_
                 scale_val = self.scaler.scale_
                 
            mean = torch.as_tensor(mean_val, device=device, dtype=dtype)
            scale = torch.as_tensor(scale_val, device=device, dtype=dtype)
            self._cov_stats_cache[key] = (mean, scale)
            return mean, scale
        return cached

    def scale_targets_tensor(
        self,
        tensor: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Scale target features in-place using torch operations."""

        if tensor.numel() == 0:
            return tensor.to(device or tensor.device)

        device = device or tensor.device
        tensor_on_device = tensor.to(device=device)
        mean, scale = self._get_target_stats_tensors(device, tensor_on_device.dtype)
        original_shape = tensor_on_device.shape
        # Use reshape instead of view for non-contiguous tensors
        reshaped = tensor_on_device.reshape(-1, mean.numel())
        scaled = (reshaped - mean) / scale
        return scaled.reshape(original_shape).contiguous()

    def scale_covariates_tensor(
        self,
        tensor: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Scale covariate features with torch operations when available."""

        if self.scaler is None or tensor.numel() == 0:
            return tensor.to(device or tensor.device)

        device = device or tensor.device
        tensor_on_device = tensor.to(device=device)
        mean, scale = self._get_cov_stats_tensors(device, tensor_on_device.dtype)
        original_shape = tensor_on_device.shape
        reshaped = tensor_on_device.reshape(-1, mean.numel())
        scaled = (reshaped - mean) / scale
        return scaled.reshape(original_shape).contiguous()

    def has_covariate_scaler(self) -> bool:
        return self.scaler is not None and len(self.covariate_features) > 0