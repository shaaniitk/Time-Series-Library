"""
Wavelet & Hierarchical Processing Module for HF Models

This module provides hierarchical decomposition and wavelet processing capabilities
for HF models by implementing external processing layers that can be combined
with any HF backbone while leveraging existing hierarchical patterns.

Key Components:
1. WaveletProcessor: Multi-resolution wavelet decomposition
2. HierarchicalProcessor: Hierarchical time series decomposition
3. FrequencyProcessor: Frequency-domain analysis
4. Integration with existing HierarchicalEnhancedAutoformer patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import List, Dict, Optional, Tuple
from argparse import Namespace
import warnings

from utils.logger import logger


class WaveletProcessor(nn.Module):
    """
    Wavelet processing for HF models following patterns from existing analysis modules.
    
    Provides multi-resolution wavelet decomposition that can be applied to
    any HF backbone without modifying its architecture.
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Wavelet configuration
        self.wavelet_type = getattr(configs, 'wavelet_type', 'db4')
        self.decomposition_levels = getattr(configs, 'decomposition_levels', 3)
        self.reconstruction_method = getattr(configs, 'reconstruction_method', 'learnable')
        
        # Input/output dimensions
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        logger.info(f"Initializing WaveletProcessor with {self.wavelet_type} wavelet, {self.decomposition_levels} levels")
        
        # Validate wavelet type
        available_wavelets = pywt.wavelist(kind='discrete')
        if self.wavelet_type not in available_wavelets:
            logger.warning(f"Wavelet {self.wavelet_type} not available, using db4")
            self.wavelet_type = 'db4'
            
        # Wavelet properties for padding calculations
        self.wavelet = pywt.Wavelet(self.wavelet_type)
        self.filter_length = len(self.wavelet.dec_lo)
        
        # Learnable reconstruction weights (following existing patterns)
        if self.reconstruction_method == 'learnable':
            self._init_learnable_reconstruction()
            
        # Component processors for different frequency bands
        self._init_component_processors()
        
    def _init_learnable_reconstruction(self):
        """Initialize learnable reconstruction following existing pattern from analysis modules"""
        
        # Learnable weights for combining wavelet coefficients
        self.reconstruction_weights = nn.Parameter(
            torch.ones(self.decomposition_levels + 1) / (self.decomposition_levels + 1)
        )
        
        # Detail coefficient processors (one for each level)
        self.detail_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_in, self.enc_in * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.enc_in * 2, self.enc_in)
            ) for _ in range(self.decomposition_levels)
        ])
        
        # Approximation coefficient processor
        self.approx_processor = nn.Sequential(
            nn.Linear(self.enc_in, self.enc_in * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.enc_in * 2, self.enc_in)
        )
        
        logger.info(f"Initialized learnable reconstruction with {self.decomposition_levels} detail processors")
        
    def _init_component_processors(self):
        """Initialize processors for different frequency components"""
        
        # Low-frequency (trend) processor
        self.trend_processor = nn.Sequential(
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=1)
        )
        
        # High-frequency (detail) processors
        self.detail_processors_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.enc_in, self.enc_in, kernel_size=2**i + 1, padding=2**i // 2),
                nn.ReLU(),
                nn.Conv1d(self.enc_in, self.enc_in, kernel_size=1)
            ) for i in range(1, self.decomposition_levels + 1)
        ])
        
        logger.info(f"Initialized component processors: 1 trend + {len(self.detail_processors_conv)} detail")
        
    def forward(self, x, process_method='decompose_reconstruct'):
        """
        Wavelet processing with multiple methods.
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            process_method: 'decompose_reconstruct', 'component_fusion', or 'frequency_attention'
            
        Returns:
            Processed tensor or dict with components
        """
        
        if process_method == 'decompose_reconstruct':
            return self._decompose_reconstruct_forward(x)
        elif process_method == 'component_fusion':
            return self._component_fusion_forward(x)
        elif process_method == 'frequency_attention':
            return self._frequency_attention_forward(x)
        else:
            logger.warning(f"Unknown process_method: {process_method}, using decompose_reconstruct")
            return self._decompose_reconstruct_forward(x)
            
    def _decompose_reconstruct_forward(self, x):
        """
        Standard wavelet decomposition and reconstruction.
        
        Following pattern from existing wavelet analysis modules.
        """
        batch_size, seq_len, features = x.shape
        processed_features = []
        
        logger.debug(f"Processing wavelet decomposition for {features} features")
        
        for feat_idx in range(features):
            # Extract single feature across all samples
            feature_data = x[:, :, feat_idx]  # [batch_size, seq_len]
            
            batch_processed = []
            
            for batch_idx in range(batch_size):
                # Single time series
                signal = feature_data[batch_idx].detach().cpu().numpy()
                
                # Wavelet decomposition
                coeffs = self._decompose_signal(signal)
                
                # Process coefficients
                processed_coeffs = self._process_coefficients(coeffs, feat_idx)
                
                # Reconstruct signal
                reconstructed = self._reconstruct_signal(processed_coeffs)
                
                batch_processed.append(reconstructed)
                
            # Stack batch results
            feature_reconstructed = torch.stack(batch_processed).to(x.device)
            processed_features.append(feature_reconstructed)
            
        # Stack features
        result = torch.stack(processed_features, dim=-1)
        
        logger.debug(f"Completed wavelet processing: {x.shape} -> {result.shape}")
        return result
        
    def _component_fusion_forward(self, x):
        """
        Process different frequency components separately and fuse.
        
        This is more computationally efficient for large sequences.
        """
        # Transpose for conv1d: [batch, features, seq_len]
        x_conv = x.transpose(1, 2)
        
        # Process trend (low-frequency)
        trend = self.trend_processor(x_conv)
        
        # Process details (high-frequency)
        details = []
        for detail_proc in self.detail_processors_conv:
            detail = detail_proc(x_conv)
            details.append(detail)
            
        # Combine components with learnable weights
        components = [trend] + details
        weights = F.softmax(self.reconstruction_weights, dim=0)
        
        fused = sum(w * comp for w, comp in zip(weights, components))
        
        # Transpose back: [batch, seq_len, features]
        result = fused.transpose(1, 2)
        
        return result
        
    def _frequency_attention_forward(self, x):
        """
        Frequency-domain attention mechanism.
        
        Applies attention to different frequency bands identified by wavelets.
        """
        batch_size, seq_len, features = x.shape
        
        # Simple implementation: decompose, attend, reconstruct
        components = self._extract_frequency_components(x)
        
        # Compute attention weights for each component
        component_embeddings = []
        for comp in components:
            # Global average pooling for each component
            embed = torch.mean(comp, dim=1, keepdim=True)  # [batch, 1, features]
            component_embeddings.append(embed)
            
        # Stack and compute attention
        embeddings = torch.cat(component_embeddings, dim=1)  # [batch, n_components, features]
        
        # Simple attention mechanism
        attention_scores = torch.softmax(torch.mean(embeddings, dim=-1, keepdim=True), dim=1)
        
        # Apply attention to components
        attended_components = [att * comp for att, comp in zip(attention_scores.unbind(1), components)]
        
        # Sum attended components
        result = sum(attended_components)
        
        return result
        
    def _decompose_signal(self, signal):
        """Decompose single signal using wavelets"""
        try:
            coeffs = pywt.wavedec(signal, self.wavelet_type, level=self.decomposition_levels, mode='symmetric')
            return coeffs
        except Exception as e:
            logger.warning(f"Wavelet decomposition failed: {e}, using original signal")
            return [signal]
            
    def _process_coefficients(self, coeffs, feat_idx):
        """Process wavelet coefficients using learnable layers"""
        if len(coeffs) <= 1:
            return coeffs
            
        processed_coeffs = []
        
        # Process approximation coefficients (last element)
        if hasattr(self, 'approx_processor'):
            approx = torch.tensor(coeffs[-1], dtype=torch.float32, device=next(self.parameters()).device)
            approx = approx.unsqueeze(0).unsqueeze(-1)  # [1, seq, 1]
            
            # Only process if we have the right dimension
            if approx.shape[-1] == 1:  # Single feature
                processed_approx = self.approx_processor(approx)
                processed_coeffs.append(processed_approx.squeeze().detach().cpu().numpy())
            else:
                processed_coeffs.append(coeffs[-1])
        else:
            processed_coeffs.append(coeffs[-1])
            
        # Process detail coefficients (all but last)
        for i, detail in enumerate(coeffs[:-1]):
            if i < len(self.detail_processors):
                detail_tensor = torch.tensor(detail, dtype=torch.float32, device=next(self.parameters()).device)
                detail_tensor = detail_tensor.unsqueeze(0).unsqueeze(-1)
                
                if detail_tensor.shape[-1] == 1:
                    processed_detail = self.detail_processors[i](detail_tensor)
                    processed_coeffs.insert(i, processed_detail.squeeze().detach().cpu().numpy())
                else:
                    processed_coeffs.insert(i, detail)
            else:
                processed_coeffs.insert(i, detail)
                
        return processed_coeffs
        
    def _reconstruct_signal(self, coeffs):
        """Reconstruct signal from processed coefficients"""
        try:
            reconstructed = pywt.waverec(coeffs, self.wavelet_type, mode='symmetric')
            return torch.tensor(reconstructed, dtype=torch.float32)
        except Exception as e:
            logger.warning(f"Wavelet reconstruction failed: {e}, using first coefficient")
            return torch.tensor(coeffs[0], dtype=torch.float32)
            
    def _extract_frequency_components(self, x):
        """Extract frequency components for attention mechanism"""
        batch_size, seq_len, features = x.shape
        components = []
        
        # Use conv layers to extract different frequency bands
        x_conv = x.transpose(1, 2)  # [batch, features, seq_len]
        
        # Low frequency (trend)
        trend = self.trend_processor(x_conv).transpose(1, 2)
        components.append(trend)
        
        # High frequencies (details)
        for detail_proc in self.detail_processors_conv:
            detail = detail_proc(x_conv).transpose(1, 2)
            components.append(detail)
            
        return components
        
    def get_wavelet_info(self):
        """Get information about wavelet configuration"""
        return {
            'wavelet_type': self.wavelet_type,
            'decomposition_levels': self.decomposition_levels,
            'reconstruction_method': self.reconstruction_method,
            'filter_length': self.filter_length,
            'learnable_weights': hasattr(self, 'reconstruction_weights')
        }


class HierarchicalProcessor(nn.Module):
    """
    Hierarchical processing for HF models following patterns from HierarchicalEnhancedAutoformer.
    
    Provides multi-scale temporal modeling without modifying HF backbone.
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Hierarchical configuration
        self.hierarchy_levels = getattr(configs, 'hierarchy_levels', [1, 2, 4])
        self.aggregation_method = getattr(configs, 'aggregation_method', 'adaptive')
        self.level_weights_learnable = getattr(configs, 'level_weights_learnable', True)
        
        # Model dimensions
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, 'd_model', 512)
        
        logger.info(f"Initializing HierarchicalProcessor with levels: {self.hierarchy_levels}")
        
        # Level processors
        self._init_level_processors()
        
        # Aggregation mechanism
        self._init_aggregation()
        
    def _init_level_processors(self):
        """Initialize processors for each hierarchy level"""
        
        self.level_processors = nn.ModuleDict()
        
        for level in self.hierarchy_levels:
            # Each level has its own processing chain
            self.level_processors[f'level_{level}'] = nn.Sequential(
                nn.Linear(self.enc_in, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.enc_in)
            )
            
        logger.info(f"Created {len(self.level_processors)} level processors")
        
    def _init_aggregation(self):
        """Initialize aggregation mechanism following existing patterns"""
        
        n_levels = len(self.hierarchy_levels)
        
        if self.aggregation_method == 'adaptive':
            # Learnable aggregation weights
            self.level_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
            
            # Adaptive attention for levels
            self.level_attention = nn.Sequential(
                nn.Linear(self.enc_in * n_levels, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, n_levels),
                nn.Softmax(dim=-1)
            )
            
        elif self.aggregation_method == 'concat':
            # Concatenation with projection
            self.concat_projection = nn.Linear(self.enc_in * n_levels, self.enc_in)
            
        elif self.aggregation_method == 'residual':
            # Residual connections between levels
            self.residual_projections = nn.ModuleList([
                nn.Linear(self.enc_in, self.enc_in) for _ in range(n_levels - 1)
            ])
            
        logger.info(f"Initialized {self.aggregation_method} aggregation")
        
    def forward(self, x, target_level=None):
        """
        Hierarchical processing with multi-scale temporal modeling.
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            target_level: Specific level to process (None for all levels)
            
        Returns:
            Processed tensor with hierarchical information
        """
        
        if target_level is not None:
            return self._single_level_forward(x, target_level)
        else:
            return self._multi_level_forward(x)
            
    def _single_level_forward(self, x, level):
        """Process single hierarchy level"""
        
        # Downsample to target level
        downsampled = self._downsample(x, level)
        
        # Process at this level
        processor_name = f'level_{level}'
        if processor_name in self.level_processors:
            processed = self.level_processors[processor_name](downsampled)
        else:
            logger.warning(f"No processor for level {level}, using identity")
            processed = downsampled
            
        # Upsample back to original resolution
        upsampled = self._upsample(processed, level, x.shape[1])
        
        return upsampled
        
    def _multi_level_forward(self, x):
        """Process all hierarchy levels and aggregate"""
        
        level_outputs = []
        
        # Process each level
        for level in self.hierarchy_levels:
            level_output = self._single_level_forward(x, level)
            level_outputs.append(level_output)
            
        # Aggregate levels
        if self.aggregation_method == 'adaptive':
            return self._adaptive_aggregation(level_outputs, x)
        elif self.aggregation_method == 'concat':
            return self._concat_aggregation(level_outputs)
        elif self.aggregation_method == 'residual':
            return self._residual_aggregation(level_outputs, x)
        else:
            # Simple weighted average
            weights = F.softmax(self.level_weights, dim=0)
            aggregated = sum(w * output for w, output in zip(weights, level_outputs))
            return aggregated
            
    def _downsample(self, x, factor):
        """Downsample sequence by given factor"""
        if factor == 1:
            return x
            
        # Simple downsampling by striding
        downsampled = x[:, ::factor, :]
        
        # Ensure minimum length
        if downsampled.shape[1] < 1:
            downsampled = x[:, :1, :]
            
        return downsampled
        
    def _upsample(self, x, factor, target_length):
        """Upsample sequence to target length"""
        if factor == 1 and x.shape[1] == target_length:
            return x
            
        # Interpolate to target length
        x_permuted = x.permute(0, 2, 1)  # [batch, features, seq_len]
        upsampled = F.interpolate(x_permuted, size=target_length, mode='linear', align_corners=False)
        upsampled = upsampled.permute(0, 2, 1)  # [batch, seq_len, features]
        
        return upsampled
        
    def _adaptive_aggregation(self, level_outputs, original_x):
        """Adaptive aggregation with attention"""
        
        # Concatenate level outputs for attention computation
        concat_outputs = torch.cat(level_outputs, dim=-1)
        
        # Compute attention weights
        attention_weights = self.level_attention(concat_outputs)  # [batch, seq_len, n_levels]
        
        # Apply attention to level outputs
        aggregated = torch.zeros_like(level_outputs[0])
        
        for i, level_output in enumerate(level_outputs):
            weight = attention_weights[:, :, i:i+1]  # [batch, seq_len, 1]
            aggregated += weight * level_output
            
        return aggregated
        
    def _concat_aggregation(self, level_outputs):
        """Concatenation aggregation"""
        concatenated = torch.cat(level_outputs, dim=-1)
        projected = self.concat_projection(concatenated)
        return projected
        
    def _residual_aggregation(self, level_outputs, original_x):
        """Residual aggregation between levels"""
        
        result = level_outputs[0]
        
        for i in range(1, len(level_outputs)):
            # Residual connection between levels
            residual = self.residual_projections[i-1](result)
            result = residual + level_outputs[i]
            
        return result
        
    def get_hierarchical_info(self):
        """Get information about hierarchical configuration"""
        return {
            'hierarchy_levels': self.hierarchy_levels,
            'aggregation_method': self.aggregation_method,
            'level_weights_learnable': self.level_weights_learnable,
            'n_level_processors': len(self.level_processors)
        }


# Factory functions for creating processors
def create_wavelet_processor(configs):
    """Factory function to create wavelet processor"""
    return WaveletProcessor(configs)


def create_hierarchical_processor(configs):
    """Factory function to create hierarchical processor"""
    return HierarchicalProcessor(configs)
