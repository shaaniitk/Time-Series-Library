import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

# ... (imports and class definition remain the same) ...

class ScalerManager:
    def __init__(self, target_features: List[str], covariate_features: List[str]):
        self.target_features = target_features
        # Filter out 'date' from covariate features to avoid scaling issues
        self.covariate_features = [col for col in covariate_features if col != 'date']
        
        # Print debug info
        print(f"ScalerManager initialized with {len(self.target_features)} target features and {len(self.covariate_features)} covariate features")
        
        self.target_scaler = StandardScaler()
        self.scaler: Optional[StandardScaler] = StandardScaler() if self.covariate_features else None # Renamed for consistency

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