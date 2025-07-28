import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from utils.logger import logger

from layers.EnhancedTargetProcessor import EnhancedTargetProcessor
from layers.CovariateProcessor import CovariateProcessor
from layers.DualCrossAttention import DualCrossAttention
from layers.SequentialDecoder import SequentialDecoder
from layers.GatedMoEFFN import GatedMoEFFN
from layers.Embed import DataEmbedding_wo_pos


class ImprovedMambaHierarchical(nn.Module):
    """
    Improved MambaHierarchical with:
    1. Explicit trend-seasonal decomposition in target processing
    2. Proper sequential decoding with trend-seasonal reconstruction
    3. Better integration of decoder inputs
    4. Enhanced context fusion
    """
    
    def __init__(self, configs):
        super(ImprovedMambaHierarchical, self).__init__()
        
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
        
        # Enhanced target processing with explicit trend-seasonal decomposition
        self.target_processor = EnhancedTargetProcessor(
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
            trend_kernel_size=getattr(self.configs, 'trend_kernel_size', 25)
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
            covariate_families=getattr(self, 'covariate_families', None)
        )
        
        # Dual cross-attention for context fusion
        self.dual_cross_attention = DualCrossAttention(
            d_model=self.d_model,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout,
            use_residual=True,
            use_layer_norm=True
        )
        
        # Adaptive fusion gate for combining attended contexts
        self.adaptive_fusion_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        logger.info("Initialized Adaptive Context Fusion Gate")
        
        # Sequential decoder with trend-seasonal reconstruction
        self.sequential_decoder = SequentialDecoder(
            d_model=self.d_model,
            c_out=self.c_out,
            pred_len=self.pred_len,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            dropout=self.dropout,
            use_autoregressive=getattr(self.configs, 'use_autoregressive', True)
        )
        
        # Mixture of Experts (optional) - applied to context before decoding
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
            self.mixture_of_experts = None
        
        # Layer normalization
        self.context_norm = nn.LayerNorm(self.d_model)

        # Fallback projection layers (defined once)
        self.target_fallback_projection = nn.Linear(self.num_targets, self.d_model)
        self.covariate_fallback_projection = nn.Linear(self.num_covariates, self.d_model)
        
        # Log comprehensive configuration
        self._log_configuration()
    
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
            self.num_families = len(self.covariate_families)
        else:
            self.num_families = self.num_covariates // self.covariate_family_size
            self.covariate_families = [self.covariate_family_size] * self.num_families
        
        # Wavelet decomposition parameters
        self.wavelet_type = getattr(self.configs, 'wavelet_type', 'db4')
        self.wavelet_levels = getattr(self.configs, 'wavelet_levels', 3)
        
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
        
        # Validation
        assert self.enc_in >= self.num_targets + self.num_covariates, \
            f"enc_in ({self.enc_in}) must be >= num_targets + num_covariates ({self.num_targets + self.num_covariates})"
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.covariate_families is None and self.num_covariates % self.covariate_family_size != 0:
            logger.warning(f"Adjusting num_covariates from {self.num_covariates} to {(self.num_covariates // self.covariate_family_size) * self.covariate_family_size}")
            self.num_covariates = (self.num_covariates // self.covariate_family_size) * self.covariate_family_size
        
        if self.d_model % self.attention_heads != 0:
            logger.warning(f"d_model ({self.d_model}) not divisible by attention_heads ({self.attention_heads})")
    
    def _log_configuration(self):
        """Log comprehensive configuration."""
        config_dump = {
            'model_type': 'ImprovedMambaHierarchical',
            'improvements': [
                'Explicit trend-seasonal decomposition',
                'Sequential decoder with autoregressive generation',
                'Enhanced context fusion',
                'Proper decoder input integration'
            ],
            'basic_params': {
                'seq_len': self.seq_len, 'pred_len': self.pred_len, 'label_len': self.label_len,
                'enc_in': self.enc_in, 'dec_in': self.dec_in, 'c_out': self.c_out, 'd_model': self.d_model
            },
            'target_covariate_split': {
                'num_targets': self.num_targets, 'num_covariates': self.num_covariates,
                'covariate_families': self.covariate_families
            },
            'decomposition_params': {
                'wavelet_type': self.wavelet_type, 'wavelet_levels': self.wavelet_levels,
                'explicit_trend_seasonal': True
            },
            'mamba_params': {
                'd_state': self.mamba_d_state, 'd_conv': self.mamba_d_conv, 'expand': self.mamba_expand
            },
            'fusion_params': {
                'fusion_strategy': self.fusion_strategy, 'use_moe': self.use_moe,
                'num_experts': self.num_experts if self.use_moe else 0
            }
        }
        logger.info(f"ImprovedMambaHierarchical Configuration: {config_dump}")
    
    def _split_input_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input tensor into target and covariate components."""
        batch_size, seq_len, total_features = x.shape
        
        targets = x[:, :, :self.num_targets]
        
        if total_features >= self.num_targets + self.num_covariates:
            covariates = x[:, :, self.num_targets:self.num_targets + self.num_covariates]
        else:
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
        
        return targets, covariates
    
    def _extract_initial_values(self, x_dec: torch.Tensor) -> torch.Tensor:
        """Extract initial values for autoregressive generation from decoder input."""
        # Use the last target values from decoder input as initial values
        targets_from_dec = x_dec[:, -1, :self.num_targets]  # [B, num_targets]
        
        # If we need c_out values but only have num_targets, handle the difference
        if self.c_out != self.num_targets:
            if self.c_out > self.num_targets:
                # Pad with zeros
                padding = torch.zeros(targets_from_dec.size(0), self.c_out - self.num_targets,
                                    device=targets_from_dec.device, dtype=targets_from_dec.dtype)
                initial_values = torch.cat([targets_from_dec, padding], dim=-1)
            else:
                # Truncate
                initial_values = targets_from_dec[:, :self.c_out]
        else:
            initial_values = targets_from_dec
        
        return initial_values
    
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
        Improved forward pass with proper trend-seasonal decomposition and sequential decoding.
        """
        batch_size, seq_len, _ = x_enc.shape
        
        # Comprehensive logging
        forward_dump = {
            'input_shapes': {
                'x_enc': list(x_enc.shape), 'x_dec': list(x_dec.shape),
                'x_mark_enc': list(x_mark_enc.shape), 'x_mark_dec': list(x_mark_dec.shape)
            },
            'future_covariates': future_covariates is not None,
            'improvements_active': ['trend_seasonal_decomp', 'sequential_decoder', 'proper_decoder_integration']
        }
        logger.info(f"ImprovedMambaHierarchical Forward: {forward_dump}")
        
        # Step 1: Split input features
        targets, covariates = self._split_input_features(x_enc)
        
        # Step 2: Enhanced target processing with trend-seasonal decomposition
        try:
            target_outputs = self.target_processor(targets, enc_self_mask)
            target_context = target_outputs['fused_context']  # [B, D]
            
            target_dump = {
                'input_shape': list(targets.shape),
                'trend_context_shape': list(target_outputs['trend_context'].shape),
                'seasonal_context_shape': list(target_outputs['seasonal_context'].shape),
                'fused_context_shape': list(target_context.shape),
                'processing_status': 'success'
            }
            logger.info(f"Enhanced Target Processing: {target_dump}")
            
        except Exception as e:
            logger.error(f"Enhanced target processing failed: {e}")
            # Fallback
            target_context = self.target_fallback_projection(targets.mean(dim=1))
        
        # Step 3: Covariate processing (unchanged)
        try:
            covariate_context = self.covariate_processor(
                covariates, enc_self_mask, future_covariates
            )
            
            covariate_dump = {
                'input_shape': list(covariates.shape),
                'output_shape': list(covariate_context.shape),
                'num_families': len(self.covariate_families),
                'family_sizes': self.covariate_families,
                'processing_status': 'success'
            }
            logger.info(f"Covariate Processing: {covariate_dump}")
            
        except Exception as e:
            logger.error(f"Covariate processing failed: {e}")
            # Fallback
            covariate_context = covariates.mean(dim=1)  # [B, num_covariates]
            covariate_context = self.covariate_fallback_projection(covariate_context)
        
        # Step 4: Dual cross-attention fusion
        try:
            fused_context, attended_target, attended_covariate = self.dual_cross_attention(
                target_context, covariate_context, enc_self_mask, enc_self_mask
            )
            
            fusion_dump = {
                'target_context_shape': list(target_context.shape),
                'covariate_context_shape': list(covariate_context.shape),
                'fused_context_shape': list(fused_context.shape),
                'processing_status': 'success'
            }
            logger.info(f"Dual Cross-Attention Fusion: {fusion_dump}")
            
        except Exception as e:
            logger.error(f"Dual cross-attention failed: {e}")
            # Fallback for attended contexts if cross-attention fails
            attended_target, attended_covariate = target_context, covariate_context

        # Combine attended contexts using the adaptive fusion gate
        try:
            gate = self.adaptive_fusion_gate(torch.cat([attended_target, attended_covariate], dim=-1))
            final_context = gate * attended_target + (1 - gate) * attended_covariate
        except Exception as e:
            logger.error(f"Adaptive context fusion failed: {e}")
            final_context = (attended_target + attended_covariate) / 2
        
        # Step 5: Apply MoE to the single, fused context (if enabled)
        if self.mixture_of_experts is not None:
            try:
                enhanced_context = self.mixture_of_experts(final_context)
                
                if isinstance(enhanced_context, tuple):
                    enhanced_context, aux_loss = enhanced_context
                    self._last_aux_loss = aux_loss
                
                logger.debug(f"MoE enhanced context shape: {enhanced_context.shape}")
                
            except Exception as e:
                logger.error(f"MoE processing failed: {e}")
                enhanced_context = final_context
        else:
            enhanced_context = final_context
        
        # Step 6: Extract initial values from decoder input
        initial_values = self._extract_initial_values(x_dec)
        
        # Step 7: Sequential decoding with trend-seasonal reconstruction
        try:
            decoder_outputs = self.sequential_decoder(
                context=enhanced_context,
                initial_values=initial_values,
                future_covariates=future_covariates
            )
            
            final_output = decoder_outputs['final']  # [B, pred_len, c_out]
            
            decoder_dump = {
                'initial_values_shape': list(initial_values.shape),
                'final_output_shape': list(final_output.shape),
                'processing_status': 'success'
            }
            logger.info(f"Sequential Decoder: {decoder_dump}")
            
        except Exception as e:
            logger.error(f"Sequential decoding failed: {e}")
            # Fallback: simple projection and repeat from the enhanced context
            projected = nn.Linear(self.d_model, self.c_out).to(enhanced_context.device)(enhanced_context)
            final_output = projected.unsqueeze(1).repeat(1, self.pred_len, 1)
        
        # Final output logging
        final_dump = {
            'final_output_shape': list(final_output.shape),
            'expected_shape': [batch_size, self.pred_len, self.c_out],
            'shape_match': list(final_output.shape) == [batch_size, self.pred_len, self.c_out]
        }
        logger.info(f"Final Output: {final_dump}")
        
        return final_output
    
    def get_auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Return auxiliary loss from MoE if available."""
        # With a single MoE, we only have one potential auxiliary loss.
        return getattr(self, '_last_aux_loss', None)
    
    def get_decomposition_outputs(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed intermediate outputs for analysis using forward hooks.
        This is a more powerful and flexible way to inspect the model's state.

        Example modules to inspect: 'target_processor', 'covariate_processor', 
                                   'dual_cross_attention', 'mixture_of_experts'
        """
        self.eval()  # Ensure model is in eval mode
        intermediate_outputs = {}

        # Define a hook function
        def get_hook(name):
            def hook(model, input, output):
                intermediate_outputs[name] = output
            return hook

        # Register hooks to desired modules
        hooks = []
        modules_to_inspect = {
            'target_processor': self.target_processor,
            'covariate_processor': self.covariate_processor,
            'dual_cross_attention': self.dual_cross_attention,
            'mixture_of_experts': self.mixture_of_experts
        }

        for name, module in modules_to_inspect.items():
            if module is not None:
                hooks.append(module.register_forward_hook(get_hook(name)))

        # Run a forward pass to trigger the hooks
        with torch.no_grad():
            _ = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Remove the hooks to avoid affecting subsequent passes
        for hook in hooks:
            hook.remove()

        # Post-process outputs for clarity
        if 'dual_cross_attention' in intermediate_outputs:
            # Unpack the tuple output from DualCrossAttention
            fused, attended_t, attended_c = intermediate_outputs['dual_cross_attention']
            intermediate_outputs['fused_context_from_attention'] = fused
            intermediate_outputs['attended_target_context'] = attended_t
            intermediate_outputs['attended_covariate_context'] = attended_c
            del intermediate_outputs['dual_cross_attention']

        return intermediate_outputs


# Alias for consistency
Model = ImprovedMambaHierarchical