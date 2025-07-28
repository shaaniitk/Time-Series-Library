"""
MambaHierarchical - Main model combining target and covariate processing with Mamba blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from utils.logger import logger

from layers.MambaBlock import MambaBlock, TargetMambaBlock, CovariateMambaBlock
from layers.TargetProcessor import TargetProcessor
from layers.CovariateProcessor import CovariateProcessor
from layers.DualCrossAttention import DualCrossAttention
from layers.GatedMoEFFN import GatedMoEFFN
from layers.Embed import DataEmbedding_wo_pos


class MambaHierarchical(nn.Module):
    """
    MambaHierarchical model that processes targets and covariates through separate pathways:
    
    Target Path: Wavelet Decomposition → Mamba → Multi-Head Attention → Context Vector
    Covariate Path: Family Grouping → Mamba Blocks → Hierarchical Attention → Context Vector
    Fusion: Dual Cross-Attention → Mixture of Experts → Output
    """
    
    def __init__(self, configs):
        super(MambaHierarchical, self).__init__()
        
        self.configs = configs
        self._extract_config_params()
        self._validate_config()
        
        # Input embeddings
        self.enc_embedding = DataEmbedding_wo_pos(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            self.dec_in, self.d_model, self.embed, self.freq, self.dropout
        )
        
        # Target processing pipeline
        self.target_processor = TargetProcessor(
            num_targets=self.num_targets,
            seq_len=self.seq_len,
            d_model=self.d_model,
            wavelet_type=self.wavelet_type,
            wavelet_levels=self.wavelet_levels,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            attention_heads=self.attention_heads,
            dropout=self.dropout,
            use_trend_decomposition=self.use_trend_decomposition
        )
        
        # Covariate processing pipeline
        self.covariate_processor = CovariateProcessor(
            num_covariates=self.num_covariates,
            family_size=self.covariate_family_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            hierarchical_attention_heads=self.hierarchical_attention_heads,
            dropout=self.dropout,
            use_family_attention=self.use_family_attention,
            fusion_strategy=self.fusion_strategy,
            covariate_families=self.covariate_families if hasattr(self, 'covariate_families') else None
        )
        
        # Dual cross-attention for context fusion
        self.dual_cross_attention = DualCrossAttention(
            d_model=self.d_model,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout,
            use_residual=True,
            use_layer_norm=True
        )
        
        # Mixture of Experts (optional)
        if self.use_moe:
            try:
                self.mixture_of_experts = GatedMoEFFN(
                    d_model=self.d_model,
                    d_ff=self.d_model * 4,
                    num_experts=self.num_experts,
                    dropout=self.dropout
                )
                logger.info(f"Initialized MoE with {self.num_experts} experts")
            except Exception as e:
                logger.warning(f"Failed to initialize MoE: {e}, using standard FFN")
                self.mixture_of_experts = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model * 4, self.d_model)
                )
        else:
            # Standard feed-forward network
            self.mixture_of_experts = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 4, self.d_model)
            )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.c_out)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Comprehensive initialization logging
        logger.info(f"MambaHierarchical initialized: {self.num_targets} targets, "
                   f"{self.num_covariates} covariates, d_model={self.d_model}")
        
        # Dump complete configuration
        config_dump = {
            'model_type': 'MambaHierarchical',
            'basic_params': {
                'seq_len': self.seq_len, 'pred_len': self.pred_len, 'label_len': self.label_len,
                'enc_in': self.enc_in, 'dec_in': self.dec_in, 'c_out': self.c_out, 'd_model': self.d_model
            },
            'target_covariate_split': {
                'num_targets': self.num_targets, 'num_covariates': self.num_covariates,
                'covariate_family_size': self.covariate_family_size,
                'num_families': self.num_covariates // self.covariate_family_size
            },
            'mamba_params': {
                'd_state': self.mamba_d_state, 'd_conv': self.mamba_d_conv, 'expand': self.mamba_expand
            },
            'attention_params': {
                'attention_heads': self.attention_heads,
                'hierarchical_attention_heads': self.hierarchical_attention_heads,
                'cross_attention_heads': self.cross_attention_heads
            },
            'wavelet_params': {
                'wavelet_type': self.wavelet_type, 'wavelet_levels': self.wavelet_levels,
                'use_trend_decomposition': self.use_trend_decomposition
            },
            'fusion_params': {
                'fusion_strategy': self.fusion_strategy, 'use_family_attention': self.use_family_attention,
                'use_moe': self.use_moe, 'num_experts': self.num_experts if self.use_moe else 0
            }
        }
        logger.info(f"MambaHierarchical Configuration Dump: {config_dump}")
    
    def _extract_config_params(self):
        """Extract and set parameters from config with defaults."""
        # Basic model parameters
        self.task_name = getattr(self.configs, 'task_name', 'long_term_forecast')
        self.seq_len = self.configs.seq_len
        self.label_len = self.configs.label_len
        self.pred_len = self.configs.pred_len
        self.enc_in = self.configs.enc_in
        self.dec_in = self.configs.dec_in
        self.c_out = self.configs.c_out
        self.d_model = self.configs.d_model
        self.dropout = getattr(self.configs, 'dropout', 0.1)
        self.embed = getattr(self.configs, 'embed', 'timeF')
        self.freq = getattr(self.configs, 'freq', 'h')
        
        # Target and covariate configuration
        self.num_targets = getattr(self.configs, 'num_targets', 4)
        self.num_covariates = getattr(self.configs, 'num_covariates', 40)
        self.covariate_family_size = getattr(self.configs, 'covariate_family_size', 4)
        
        # Support for custom covariate family configurations
        self.covariate_families = getattr(self.configs, 'covariate_families', None)
        if self.covariate_families is not None:
            # Custom family configuration provided
            self.num_families = len(self.covariate_families)
            logger.info(f"Using custom covariate families: {self.covariate_families}")
        else:
            # Default uniform family configuration
            self.num_families = self.num_covariates // self.covariate_family_size
            self.covariate_families = [self.covariate_family_size] * self.num_families
            logger.info(f"Using uniform covariate families: {self.num_families} families of size {self.covariate_family_size}")
        
        # Wavelet decomposition parameters
        self.wavelet_type = getattr(self.configs, 'wavelet_type', 'db4')
        self.wavelet_levels = getattr(self.configs, 'wavelet_levels', 3)
        self.use_trend_decomposition = getattr(self.configs, 'use_trend_decomposition', False)
        
        # Mamba parameters
        self.mamba_d_state = getattr(self.configs, 'mamba_d_state', 64)
        self.mamba_d_conv = getattr(self.configs, 'mamba_d_conv', 4)
        self.mamba_expand = getattr(self.configs, 'mamba_expand', 2)
        
        # Attention parameters
        self.attention_heads = getattr(self.configs, 'attention_heads', 8)
        self.hierarchical_attention_heads = getattr(self.configs, 'hierarchical_attention_heads', 8)
        self.cross_attention_heads = getattr(self.configs, 'cross_attention_heads', 8)
        
        # Fusion and MoE parameters
        self.fusion_strategy = getattr(self.configs, 'fusion_strategy', 'weighted_concat')
        self.use_family_attention = getattr(self.configs, 'use_family_attention', True)
        self.use_moe = getattr(self.configs, 'use_moe', True)
        self.num_experts = getattr(self.configs, 'num_experts', 8)
        
        # Validation parameters
        assert self.enc_in >= self.num_targets + self.num_covariates, \
            f"enc_in ({self.enc_in}) must be >= num_targets + num_covariates ({self.num_targets + self.num_covariates})"
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.num_covariates % self.covariate_family_size != 0:
            logger.warning(f"num_covariates ({self.num_covariates}) not divisible by "
                          f"covariate_family_size ({self.covariate_family_size}). "
                          f"Adjusting num_covariates to {(self.num_covariates // self.covariate_family_size) * self.covariate_family_size}")
            self.num_covariates = (self.num_covariates // self.covariate_family_size) * self.covariate_family_size
        
        if self.d_model % self.attention_heads != 0:
            logger.warning(f"d_model ({self.d_model}) not divisible by attention_heads ({self.attention_heads})")
    
    def _split_input_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split input tensor into target and covariate components.
        
        Args:
            x: Input tensor [batch_size, seq_len, total_features]
            
        Returns:
            Tuple of (targets, covariates)
        """
        batch_size, seq_len, total_features = x.shape
        
        # Extract targets (first num_targets features)
        targets = x[:, :, :self.num_targets]
        
        # Extract covariates (next num_covariates features)
        if total_features >= self.num_targets + self.num_covariates:
            covariates = x[:, :, self.num_targets:self.num_targets + self.num_covariates]
        else:
            # Pad with zeros if not enough features
            available_covariates = total_features - self.num_targets
            if available_covariates > 0:
                existing_covariates = x[:, :, self.num_targets:]
                padding = torch.zeros(batch_size, seq_len, 
                                    self.num_covariates - available_covariates,
                                    device=x.device, dtype=x.dtype)
                covariates = torch.cat([existing_covariates, padding], dim=-1)
            else:
                covariates = torch.zeros(batch_size, seq_len, self.num_covariates,
                                       device=x.device, dtype=x.dtype)
        
        logger.debug(f"Split features: targets {targets.shape}, covariates {covariates.shape}")
        return targets, covariates
    
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through MambaHierarchical model.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, enc_in]
            x_mark_enc: Encoder time features
            x_dec: Decoder input [batch_size, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features
            enc_self_mask: Optional encoder self-attention mask
            dec_self_mask: Optional decoder self-attention mask
            dec_enc_mask: Optional decoder-encoder attention mask
            future_covariates: Optional future covariate data (Hilbert-transformed)
            
        Returns:
            Output predictions [batch_size, pred_len, c_out]
        """
        batch_size, seq_len, _ = x_enc.shape
        
        # Comprehensive forward pass logging
        forward_dump = {
            'input_shapes': {
                'x_enc': list(x_enc.shape), 'x_dec': list(x_dec.shape),
                'x_mark_enc': list(x_mark_enc.shape), 'x_mark_dec': list(x_mark_dec.shape)
            },
            'batch_info': {'batch_size': batch_size, 'seq_len': seq_len},
            'masks': {
                'enc_self_mask': enc_self_mask is not None,
                'dec_self_mask': dec_self_mask is not None,
                'dec_enc_mask': dec_enc_mask is not None
            },
            'future_covariates': future_covariates is not None
        }
        logger.info(f"MambaHierarchical Forward Pass Dump: {forward_dump}")
        logger.debug(f"MambaHierarchical forward: x_enc {x_enc.shape}, x_dec {x_dec.shape}")
        
        # Step 1: Split input features into targets and covariates
        targets, covariates = self._split_input_features(x_enc)
        
        # Step 2: Process targets through target pipeline
        try:
            target_context = self.target_processor(targets, enc_self_mask)
            target_dump = {
                'input_shape': list(targets.shape),
                'output_shape': list(target_context.shape),
                'processing_status': 'success'
            }
            logger.info(f"Target Processing Dump: {target_dump}")
            logger.debug(f"Target context shape: {target_context.shape}")
        except Exception as e:
            logger.error(f"Target processing failed: {e}")
            # Fallback: use mean pooling
            target_context = targets.mean(dim=1)  # [B, num_targets]
            target_context = nn.Linear(self.num_targets, self.d_model).to(targets.device)(target_context)
        
        # Step 3: Process covariates through covariate pipeline
        try:
            covariate_context = self.covariate_processor(
                covariates, enc_self_mask, future_covariates
            )
            covariate_dump = {
                'input_shape': list(covariates.shape),
                'output_shape': list(covariate_context.shape),
                'num_families': self.num_covariates // self.covariate_family_size,
                'family_size': self.covariate_family_size,
                'has_future_covariates': future_covariates is not None,
                'processing_status': 'success'
            }
            logger.info(f"Covariate Processing Dump: {covariate_dump}")
            logger.debug(f"Covariate context shape: {covariate_context.shape}")
        except Exception as e:
            logger.error(f"Covariate processing failed: {e}")
            # Fallback: use mean pooling
            covariate_context = covariates.mean(dim=1)  # [B, num_covariates]
            covariate_context = nn.Linear(self.num_covariates, self.d_model).to(covariates.device)(covariate_context)
        
        # Step 4: Apply dual cross-attention
        try:
            fused_context, attended_target, attended_covariate = self.dual_cross_attention(
                target_context, covariate_context, enc_self_mask, enc_self_mask
            )
            fusion_dump = {
                'target_context_shape': list(target_context.shape),
                'covariate_context_shape': list(covariate_context.shape),
                'fused_context_shape': list(fused_context.shape),
                'attended_target_shape': list(attended_target.shape),
                'attended_covariate_shape': list(attended_covariate.shape),
                'processing_status': 'success'
            }
            logger.info(f"Dual Cross-Attention Fusion Dump: {fusion_dump}")
            logger.debug(f"Fused context shape: {fused_context.shape}")
        except Exception as e:
            logger.error(f"Dual cross-attention failed: {e}")
            # Fallback: simple average
            fused_context = (target_context + covariate_context) / 2
        
        # Step 5: Apply Mixture of Experts or standard FFN
        try:
            if self.use_moe and hasattr(self.mixture_of_experts, 'forward'):
                # MoE forward pass
                expert_output = self.mixture_of_experts(fused_context)
                if isinstance(expert_output, tuple):
                    # Handle MoE returning (output, aux_loss)
                    expert_output, aux_loss = expert_output
                    # Store aux_loss for potential use in training
                    self._last_aux_loss = aux_loss
            else:
                # Standard FFN
                expert_output = self.mixture_of_experts(fused_context)
            
            logger.debug(f"Expert output shape: {expert_output.shape}")
        except Exception as e:
            logger.error(f"MoE/FFN processing failed: {e}")
            expert_output = fused_context
        
        # Step 6: Apply final normalization
        normalized_output = self.final_norm(expert_output)
        
        # Step 7: Project to output dimension
        output = self.output_projection(normalized_output)
        
        # Step 8: Expand to prediction sequence length if needed
        if output.dim() == 2:  # [B, c_out]
            # Repeat for prediction length
            output = output.unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, pred_len, c_out]
        
        # Final output dump
        final_dump = {
            'expert_output_shape': list(expert_output.shape),
            'normalized_output_shape': list(normalized_output.shape),
            'projected_output_shape': list(output.shape),
            'expected_output_shape': [batch_size, self.pred_len, self.c_out],
            'moe_used': self.use_moe,
            'has_aux_loss': hasattr(self, '_last_aux_loss') and self._last_aux_loss is not None
        }
        logger.info(f"Final Output Dump: {final_dump}")
        logger.debug(f"Final output shape: {output.shape}")
        
        # Return auxiliary loss if available for training compatibility
        aux_loss = self.get_auxiliary_loss()
        if aux_loss is not None:
            return output, aux_loss
        else:
            return output
    
    def get_auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Return auxiliary loss from MoE if available."""
        return getattr(self, '_last_aux_loss', None)
    
    def get_attention_weights(self) -> Dict[str, Any]:
        """Return attention weights from various components for visualization."""
        weights = {}
        
        # Target processor attention weights
        if hasattr(self.target_processor, 'get_attention_weights'):
            weights['target_attention'] = self.target_processor.get_attention_weights()
        
        # Dual cross-attention weights
        if hasattr(self.dual_cross_attention, 'get_attention_weights'):
            target_weights, covariate_weights = self.dual_cross_attention.get_attention_weights()
            weights['cross_attention_target'] = target_weights
            weights['cross_attention_covariate'] = covariate_weights
        
        return weights
    
    def get_component_outputs(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        future_covariates: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate outputs from all components for analysis.
        
        Returns:
            Dictionary containing outputs from each processing stage
        """
        targets, covariates = self._split_input_features(x_enc)
        
        outputs = {
            'targets': targets,
            'covariates': covariates,
            'target_context': self.target_processor(targets),
            'covariate_context': self.covariate_processor(covariates, future_covariates=future_covariates)
        }
        
        fused_context, attended_target, attended_covariate = self.dual_cross_attention(
            outputs['target_context'], outputs['covariate_context']
        )
        
        outputs.update({
            'fused_context': fused_context,
            'attended_target': attended_target,
            'attended_covariate': attended_covariate
        })
        
        return outputs


# Alias for consistency with other models
Model = MambaHierarchical