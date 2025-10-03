"""Patch for ScalerManager to fix tensor reshaping issues."""

import torch
from typing import Optional

def patched_scale_targets_tensor(self, tensor: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Scale target features in-place using torch operations with fixed reshaping."""
    
    if tensor.numel() == 0:
        return tensor.to(device or tensor.device)

    device = device or tensor.device
    tensor_on_device = tensor.to(device=device)
    mean, scale = self._get_target_stats_tensors(device, tensor_on_device.dtype)
    original_shape = tensor_on_device.shape
    
    # Use reshape instead of view to handle non-contiguous tensors
    reshaped = tensor_on_device.reshape(-1, mean.numel())
    scaled = (reshaped - mean) / scale
    return scaled.reshape(original_shape).contiguous()

def patched_scale_covariates_tensor(self, tensor: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Scale covariate features with torch operations when available with fixed reshaping."""
    
    if self.scaler is None or tensor.numel() == 0:
        return tensor.to(device or tensor.device)

    device = device or tensor.device
    tensor_on_device = tensor.to(device=device)
    mean, scale = self._get_covariate_stats_tensors(device, tensor_on_device.dtype)
    original_shape = tensor_on_device.shape
    
    # Use reshape instead of view to handle non-contiguous tensors
    reshaped = tensor_on_device.reshape(-1, mean.numel())
    scaled = (reshaped - mean) / scale
    return scaled.reshape(original_shape).contiguous()

def apply_scaler_patch():
    """Apply the patch to ScalerManager."""
    from utils.scaler_manager import ScalerManager
    
    # Monkey patch the methods
    ScalerManager.scale_targets_tensor = patched_scale_targets_tensor
    ScalerManager.scale_covariates_tensor = patched_scale_covariates_tensor
    
    print("Applied ScalerManager tensor reshaping patch")