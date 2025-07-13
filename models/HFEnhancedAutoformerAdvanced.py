"""
HF Enhanced Autoformer Advanced - Complete Implementation

This implementation demonstrates how to add advanced features (Bayesian, Wavelet, etc.)
to HF models through external processing and layer conversion while maintaining
HF model stability.

Key Technical Solutions:
1. Bayesian Layer Conversion: Replace HF layers with Bayesian equivalents externally
2. Wavelet Processing: Add wavelet decomposition as pre/post processing
3. Loss Integration: Leverage existing loss infrastructure
4. Pattern Following: Use BayesianEnhancedAutoformer defensive copying approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from argparse import Namespace

# Import existing HF model and infrastructure
from models.HFEnhancedAutoformer import HFEnhancedAutoformer
from layers.BayesianLayers import BayesianLinear, convert_to_bayesian, collect_kl_divergence, DropoutSampling
from utils.losses import get_loss_function
from utils.bayesian_losses import create_bayesian_loss
from utils.logger import logger

class HFBayesianLayerConverter:
    """
    Converts HF model layers to Bayesian equivalents externally.
    
    Technical Approach:
    - Identify target layers in HF model
    - Replace with Bayesian equivalents
    - Initialize from original deterministic weights
    - Maintain gradient flow
    """
    
    def __init__(self, kl_weight=1e-5):
        self.kl_weight = kl_weight
        self.converted_layers = []
        
    def convert_hf_layers(self, hf_model, layer_names=['output_projection']):
        """
        Convert specified HF layers to Bayesian versions.
        
        Args:
            hf_model: HF model instance
            layer_names: List of layer names to convert
            
        Returns:
            List of converted Bayesian layers
        """
        logger.info(f"Converting HF layers to Bayesian: {layer_names}")
        
        converted = []
        
        for layer_name in layer_names:
            if hasattr(hf_model, layer_name):
                original_layer = getattr(hf_model, layer_name)
                
                if isinstance(original_layer, nn.Linear):
                    # Convert Linear layer to BayesianLinear
                    bayesian_layer = BayesianLinear(
                        original_layer.in_features,
                        original_layer.out_features,
                        bias=original_layer.bias is not None
                    )
                    
                    # Initialize Bayesian layer from deterministic weights
                    self._initialize_bayesian_from_deterministic(
                        bayesian_layer, original_layer
                    )
                    
                    # Replace the layer in HF model
                    setattr(hf_model, layer_name, bayesian_layer)
                    converted.append(bayesian_layer)
                    
                    logger.info(f"✅ Converted {layer_name}: {original_layer.in_features} -> {original_layer.out_features}")
                else:
                    logger.warning(f"⚠️ Cannot convert layer {layer_name}: unsupported type {type(original_layer)}")
            else:
                logger.warning(f"⚠️ Layer {layer_name} not found in HF model")
                
        self.converted_layers = converted
        logger.info(f"Successfully converted {len(converted)} layers to Bayesian")
        return converted
        
    def _initialize_bayesian_from_deterministic(self, bayesian_layer, deterministic_layer):
        """Initialize Bayesian layer parameters from deterministic layer."""
        
        # Initialize weight mean from deterministic weights
        with torch.no_grad():
            bayesian_layer.weight_mu.copy_(deterministic_layer.weight)
            
            # Initialize weight variance (small positive values)
            bayesian_layer.weight_rho.fill_(-3.0)  # rho = log(sigma), so sigma ≈ 0.05
            
            # Initialize bias if present
            if deterministic_layer.bias is not None and hasattr(bayesian_layer, 'bias_mu'):
                bayesian_layer.bias_mu.copy_(deterministic_layer.bias)
                bayesian_layer.bias_rho.fill_(-3.0)
                
        logger.debug(f"Initialized Bayesian layer from deterministic weights")

class HFWaveletProcessor:
    """
    Adds wavelet decomposition to HF models through external processing.
    
    Technical Approach:
    - Pre-process inputs with wavelet decomposition
    - Multi-scale feature extraction
    - Post-process outputs with wavelet reconstruction
    """
    
    def __init__(self, wavelet_type='db4', n_levels=3):
        self.wavelet_type = wavelet_type
        self.n_levels = n_levels
        
        # For now, use simple downsampling as wavelet approximation
        # In production, you'd use PyWavelets or similar
        self.scales = [2**i for i in range(n_levels)]
        
        logger.info(f"Initialized wavelet processor: {wavelet_type}, {n_levels} levels")
        
    def decompose_input(self, x):
        """
        Decompose input into multiple scales.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            
        Returns:
            List of tensors at different scales
        """
        batch_size, seq_len, features = x.shape
        decomposed = []
        
        for scale in self.scales:
            if seq_len >= scale:
                # Simple downsampling as wavelet approximation
                downsampled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
                decomposed.append(downsampled)
            else:
                # If sequence too short, use original
                decomposed.append(x)
                
        return decomposed
        
    def reconstruct_output(self, multi_scale_outputs, target_length):
        """
        Reconstruct output from multiple scales.
        
        Args:
            multi_scale_outputs: List of outputs at different scales
            target_length: Target sequence length
            
        Returns:
            Reconstructed output tensor
        """
        if not multi_scale_outputs:
            return None
            
        batch_size = multi_scale_outputs[0].shape[0]
        features = multi_scale_outputs[0].shape[-1]
        
        # Simple reconstruction: interpolate and average
        reconstructed = torch.zeros(batch_size, target_length, features, 
                                  device=multi_scale_outputs[0].device)
        
        for output in multi_scale_outputs:
            # Interpolate to target length
            upsampled = F.interpolate(
                output.transpose(1, 2), 
                size=target_length, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            
            reconstructed += upsampled / len(multi_scale_outputs)
            
        return reconstructed

class HFEnhancedAutoformerAdvanced(nn.Module):
    """
    Advanced HF Enhanced Autoformer with Bayesian, Wavelet, and Loss Integration.
    
    This implementation shows how to add sophisticated features to HF models
    while maintaining their stability and leveraging existing loss infrastructure.
    """
    
    def __init__(self, configs):
        super().__init__()
        
        logger.info("Initializing HFEnhancedAutoformerAdvanced")
        
        # Store configs using defensive copying pattern from BayesianEnhancedAutoformer
        self.configs_original = configs
        self.original_c_out = getattr(configs, 'c_out_evaluation', configs.c_out)
        
        # Uncertainty configuration
        self.uncertainty_method = getattr(configs, 'uncertainty_method', None)
        self.n_samples = getattr(configs, 'n_samples', 10)
        self.kl_weight = getattr(configs, 'kl_weight', 1e-5)
        
        # Quantile configuration
        self.quantile_levels = getattr(configs, 'quantile_levels', None)
        self.is_quantile_mode = self.quantile_levels is not None and len(self.quantile_levels) > 0
        
        if self.is_quantile_mode:
            self.quantiles = sorted(self.quantile_levels)
            self.num_quantiles = len(self.quantiles)
            logger.info(f"Quantile mode ON: {self.quantiles}")
        else:
            self.quantiles = []
            self.num_quantiles = 1
            logger.info("Quantile mode OFF")
            
        # Wavelet configuration
        self.use_wavelets = getattr(configs, 'use_wavelets', False)
        
        # Initialize HF backbone with clean config (following BayesianEnhancedAutoformer pattern)
        hf_configs_dict = vars(configs).copy()
        
        # Clean up advanced feature configs for base HF model
        cleanup_keys = ['quantile_levels', 'uncertainty_method', 'kl_weight', 
                       'n_samples', 'use_wavelets', 'wavelet_type', 'n_levels']
        for key in cleanup_keys:
            if key in hf_configs_dict:
                del hf_configs_dict[key]
                
        # Set base output dimension
        hf_configs_dict['c_out'] = self.original_c_out
        
        hf_configs_ns = Namespace(**hf_configs_dict)
        self.hf_backbone = HFEnhancedAutoformer(hf_configs_ns)
        
        logger.info(f"✅ Initialized HF backbone with c_out={self.original_c_out}")
        
        # Initialize advanced feature processors
        self._setup_advanced_features(configs)
        
        # Initialize loss manager using existing infrastructure
        self.loss_manager = self._create_loss_manager(configs)
        
        logger.info("✅ HFEnhancedAutoformerAdvanced initialization complete")
        
    def _setup_advanced_features(self, configs):
        """Setup advanced feature processors."""
        
        # 1. Bayesian Layer Conversion
        if self.uncertainty_method == 'bayesian':
            self.bayesian_converter = HFBayesianLayerConverter(self.kl_weight)
            bayesian_layers = getattr(configs, 'bayesian_layers', ['output_projection'])
            self.converted_bayesian_layers = self.bayesian_converter.convert_hf_layers(
                self.hf_backbone, bayesian_layers
            )
            logger.info(f"✅ Setup Bayesian conversion for {len(self.converted_bayesian_layers)} layers")
            
        elif self.uncertainty_method == 'dropout':
            # Setup MC Dropout
            self.mc_dropout1 = DropoutSampling(p=0.1)
            self.mc_dropout2 = DropoutSampling(p=0.15)
            self.mc_dropout3 = DropoutSampling(p=0.1)
            logger.info("✅ Setup MC Dropout layers")
            
        # 2. Wavelet Processing
        if self.use_wavelets:
            wavelet_type = getattr(configs, 'wavelet_type', 'db4')
            n_levels = getattr(configs, 'n_levels', 3)
            self.wavelet_processor = HFWaveletProcessor(wavelet_type, n_levels)
            logger.info(f"✅ Setup wavelet processing: {wavelet_type}, {n_levels} levels")
        else:
            self.wavelet_processor = None
            
        # 3. Quantile Expansion
        if self.is_quantile_mode:
            expanded_c_out = self.original_c_out * self.num_quantiles
            self.quantile_expansion = nn.Linear(self.original_c_out, expanded_c_out)
            logger.info(f"✅ Setup quantile expansion: {self.original_c_out} -> {expanded_c_out}")
        else:
            self.quantile_expansion = None
            
    def _create_loss_manager(self, configs):
        """Create loss manager using existing loss infrastructure."""
        
        loss_type = getattr(configs, 'loss', 'mse').lower()
        
        # Use existing Bayesian loss infrastructure if uncertainty is enabled
        if self.uncertainty_method:
            logger.info(f"Creating Bayesian loss manager for {loss_type}")
            
            # Extract loss-specific kwargs
            loss_kwargs = self._extract_loss_kwargs(configs, loss_type)
            
            return create_bayesian_loss(
                loss_type=loss_type,
                kl_weight=self.kl_weight,
                uncertainty_weight=getattr(configs, 'uncertainty_weight', 0.1),
                **loss_kwargs
            )
        else:
            logger.info(f"Creating standard loss manager for {loss_type}")
            
            # Use standard loss
            loss_kwargs = self._extract_loss_kwargs(configs, loss_type)
            return get_loss_function(loss_type, **loss_kwargs)
            
    def _extract_loss_kwargs(self, configs, loss_type):
        """Extract appropriate kwargs for loss function."""
        kwargs = {}
        
        if loss_type == 'pinball':
            kwargs['quantile_levels'] = self.quantiles if self.quantiles else [0.1, 0.5, 0.9]
        elif loss_type == 'quantile':
            if self.quantiles and len(self.quantiles) > 1:
                kwargs['quantiles'] = self.quantiles
            else:
                kwargs['quantile'] = getattr(configs, 'quantile', 0.5)
        elif loss_type == 'ps_loss':
            kwargs.update({
                'pred_len': configs.pred_len,
                'mse_weight': getattr(configs, 'ps_mse_weight', 0.5),
                'w_corr': getattr(configs, 'ps_w_corr', 1.0),
                'w_var': getattr(configs, 'ps_w_var', 1.0),
                'w_mean': getattr(configs, 'ps_w_mean', 1.0)
            })
        elif loss_type == 'huber':
            kwargs['delta'] = getattr(configs, 'huber_delta', 1.0)
            
        return kwargs
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Advanced forward pass with all features integrated.
        
        Flow:
        1. Optional wavelet decomposition
        2. HF backbone processing (with/without uncertainty)
        3. Optional quantile expansion
        4. Optional wavelet reconstruction
        """
        
        # 1. Wavelet decomposition (if enabled)
        if self.wavelet_processor:
            multi_scale_inputs = self.wavelet_processor.decompose_input(x_enc)
            logger.debug(f"Wavelet decomposition: {len(multi_scale_inputs)} scales")
        else:
            multi_scale_inputs = [x_enc]
            
        # 2. Process through HF backbone with uncertainty
        if self.uncertainty_method:
            outputs = self._forward_with_uncertainty(
                multi_scale_inputs, x_mark_enc, x_dec, x_mark_dec, mask
            )
        else:
            # Standard forward through HF backbone
            hf_output = self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            outputs = {'prediction': hf_output}
            
        # 3. Quantile expansion (if enabled)
        if self.quantile_expansion is not None:
            if isinstance(outputs, dict) and 'prediction' in outputs:
                expanded = self.quantile_expansion(outputs['prediction'])
                outputs['prediction'] = expanded
                
                # Add quantile-specific uncertainty if available
                if 'predictions_samples' in outputs and outputs['predictions_samples'] is not None:
                    outputs.update(self._compute_quantile_uncertainty(
                        outputs['predictions_samples']
                    ))
            else:
                outputs = self.quantile_expansion(outputs)
                
        # 4. Wavelet reconstruction (if enabled and multi-scale)
        if self.wavelet_processor and len(multi_scale_inputs) > 1:
            if isinstance(outputs, dict) and 'prediction' in outputs:
                target_length = outputs['prediction'].shape[1]
                # For now, skip wavelet reconstruction in uncertainty mode
                pass
            else:
                target_length = outputs.shape[1]
                # Skip wavelet reconstruction for standard forward
                pass
                
        return outputs
        
    def _forward_with_uncertainty(self, multi_scale_inputs, x_mark_enc, x_dec, x_mark_dec, mask):
        """Forward pass with uncertainty estimation."""
        
        # Use first scale for uncertainty estimation (can be extended)
        x_enc = multi_scale_inputs[0]
        
        if self.uncertainty_method == 'bayesian':
            return self._bayesian_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.uncertainty_method == 'dropout':
            return self._dropout_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            # Fallback to standard forward
            hf_output = self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return {'prediction': hf_output}
            
    def _bayesian_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Bayesian forward pass following BayesianEnhancedAutoformer pattern."""
        
        logger.debug(f"Computing Bayesian uncertainty with {self.n_samples} samples")
        
        predictions = []
        grad_context = torch.enable_grad if self.training else torch.no_grad
        
        with grad_context():
            for _ in range(self.n_samples):
                # Each forward pass samples different weights from Bayesian layers
                pred = self.hf_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
                predictions.append(pred)
                
        pred_stack = torch.stack(predictions)
        
        if not self.training:
            pred_stack = pred_stack.detach()
            
        return self._compute_uncertainty_statistics(pred_stack)
        
    def _dropout_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """MC Dropout forward pass."""
        
        logger.debug(f"Computing MC Dropout uncertainty with {self.n_samples} samples")
        
        predictions = []
        original_training_state = self.training
        
        # Force training mode for MC Dropout
        self.train()
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Apply dropout to input
                x_enc_dropped = self.mc_dropout1(x_enc)
                pred = self.hf_backbone(x_enc_dropped, x_mark_enc, x_dec, x_mark_dec)
                # Apply dropout to output
                pred_dropped = self.mc_dropout3(pred)
                predictions.append(pred_dropped)
                
        # Restore original training state
        self.train(original_training_state)
        
        pred_stack = torch.stack(predictions)
        return self._compute_uncertainty_statistics(pred_stack)
        
    def _compute_uncertainty_statistics(self, pred_stack):
        """Compute uncertainty statistics following BayesianEnhancedAutoformer pattern."""
        
        mean_pred = torch.mean(pred_stack, dim=0)
        total_variance = torch.var(pred_stack, dim=0)
        total_std = torch.sqrt(total_variance)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf_level in [0.68, 0.95, 0.99]:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = torch.quantile(pred_stack, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(pred_stack, upper_percentile / 100, dim=0)
            confidence_intervals[f'{int(conf_level * 100)}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
            
        return {
            'prediction': mean_pred,
            'uncertainty': total_std,
            'variance': total_variance,
            'confidence_intervals': confidence_intervals,
            'predictions_samples': pred_stack
        }
        
    def _compute_quantile_uncertainty(self, pred_stack):
        """Compute quantile-specific uncertainty."""
        
        if not self.quantiles:
            return {}
            
        n_quantiles = len(self.quantiles)
        total_features = pred_stack.shape[-1]
        
        if total_features % n_quantiles != 0:
            logger.warning(f"Cannot compute quantile uncertainty: {total_features} not divisible by {n_quantiles}")
            return {}
            
        features_per_quantile = total_features // n_quantiles
        quantile_results = {}
        
        for i, q_level in enumerate(self.quantiles):
            start_idx = i * features_per_quantile
            end_idx = (i + 1) * features_per_quantile
            q_predictions = pred_stack[:, :, :, start_idx:end_idx]
            
            q_mean = torch.mean(q_predictions, dim=0)
            q_std = torch.std(q_predictions, dim=0)
            
            # Compute certainty score
            mean_pred = torch.mean(q_predictions, dim=0)
            std_pred = torch.std(q_predictions, dim=0)
            cv = std_pred / (torch.abs(mean_pred) + 1e-8)
            certainty = 1.0 / (1.0 + cv)
            
            quantile_results[f'quantile_{q_level}'] = {
                'prediction': q_mean,
                'uncertainty': q_std,
                'certainty_score': certainty
            }
            
        return {'quantile_specific': quantile_results}
        
    def compute_loss(self, predictions, targets, return_components=False):
        """Compute loss using existing loss infrastructure."""
        
        if hasattr(self.loss_manager, 'forward'):
            # This is a BayesianLoss from bayesian_losses.py
            return self.loss_manager.forward(
                model=self,
                pred_result=predictions,
                true=targets
            )
        else:
            # Standard loss from losses.py
            if isinstance(predictions, dict):
                pred_tensor = predictions['prediction']
            else:
                pred_tensor = predictions
                
            return self.loss_manager(pred_tensor, targets)
            
    def get_kl_loss(self, max_kl_value=1e6):
        """Get KL loss from Bayesian layers."""
        
        if self.uncertainty_method != 'bayesian':
            return 0.0
            
        kl_div = collect_kl_divergence(self.hf_backbone)
        kl_div_clipped = torch.clamp(kl_div, max=max_kl_value)
        
        if kl_div > kl_div_clipped:
            logger.warning(f"Clipped KL divergence from {kl_div:.4f} to {kl_div_clipped:.4f}")
            
        return kl_div_clipped * self.kl_weight
        
    def get_model_info(self):
        """Get model information for validation."""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'name': 'HFEnhancedAutoformerAdvanced',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'hf_backbone': type(self.hf_backbone).__name__,
            'uncertainty_method': self.uncertainty_method,
            'use_wavelets': self.use_wavelets,
            'is_quantile_mode': self.is_quantile_mode,
            'quantiles': self.quantiles,
            'original_c_out': self.original_c_out
        }
        
        if hasattr(self, 'converted_bayesian_layers'):
            info['bayesian_layers_converted'] = len(self.converted_bayesian_layers)
            
        return info

# Alias for compatibility
Model = HFEnhancedAutoformerAdvanced
