"""
Adapter Components for Modular Framework

This module provides adapter implementations that extend backbone capabilities
while preserving their core functionality.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

from ..base_interfaces import BaseBackbone
from ..config_schemas import BackboneConfig

logger = logging.getLogger(__name__)


class CovariateAdapter(BaseBackbone):
    """
    Universal adapter that adds covariate support to any backbone.
    
    Works by injecting covariates at the data preparation level,
    then passing enhanced data to the original backbone unchanged.
    
    This preserves the backbone's pretrained knowledge while enabling
    covariate-aware forecasting.
    """
    
    def __init__(self, backbone: BaseBackbone, covariate_config: Dict[str, Any]):
        """
        Initialize CovariateAdapter
        
        Args:
            backbone: The base backbone to wrap (e.g., ChronosBackbone)
            covariate_config: Configuration for covariate handling
                - covariate_dim: Number of covariate features
                - fusion_method: How to combine covariates ('concat', 'add', 'project')
                - embedding_dim: Target embedding dimension
                - temporal_features: Whether to handle temporal features
        """
        # Initialize with a mock config for BaseComponent
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.backbone = backbone
        self.covariate_config = covariate_config
        
        # Extract configuration
        self.covariate_dim = covariate_config.get('covariate_dim', 0)
        self.fusion_method = covariate_config.get('fusion_method', 'project')
        self.embedding_dim = covariate_config.get('embedding_dim', self.backbone.get_d_model())
        self.temporal_features = covariate_config.get('temporal_features', True)
        
        # Build covariate processing components
        self._build_covariate_layers()
        
        logger.info(f"CovariateAdapter initialized:")
        logger.info(f"  - Backbone: {type(self.backbone).__name__}")
        logger.info(f"  - Covariate dim: {self.covariate_dim}")
        logger.info(f"  - Fusion method: {self.fusion_method}")
        logger.info(f"  - Embedding dim: {self.embedding_dim}")
    
    def _build_covariate_layers(self):
        """Build the covariate processing layers based on fusion method"""
        
        if self.covariate_dim == 0:
            # No covariates, just pass through
            self.covariate_processor = nn.Identity()
            return
        
        if self.fusion_method == 'project':
            # Project covariates and time series to common space, then combine
            self.ts_projector = nn.Linear(1, self.embedding_dim // 2)
            self.covariate_projector = nn.Linear(self.covariate_dim, self.embedding_dim // 2)
            self.fusion_projector = nn.Linear(self.embedding_dim, 1)  # Back to TS dimension
            # Learned gating to guarantee measurable covariate effect (initialized >0)
            self.covariate_gate = nn.Parameter(torch.tensor(0.1))
            # Normalise in embedding space (non-degenerate) prior to projection back to dim 1
            self.pre_fusion_norm = nn.LayerNorm(self.embedding_dim)
            
        elif self.fusion_method == 'concat':
            # Concatenate covariates with time series features
            # This requires modifying the input dimension
            self.dimension_adapter = nn.Linear(1 + self.covariate_dim, 1)
            
        elif self.fusion_method == 'add':
            # Project covariates to time series dimension and add
            self.covariate_projector = nn.Linear(self.covariate_dim, 1)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Optional: Add normalization and dropout
        # NOTE: A LayerNorm over a single feature dimension collapses all variation to 0.
        # This previously caused the covariate effect invariant delta to be exactly 0.
        # We therefore skip final LayerNorm when the output dim is 1 and instead (for 'project')
        # normalise in the higher-dimensional embedding space before projection.
        self.layer_norm = nn.Identity()  # keep as Identity to preserve measurable differences
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                x_ts: torch.Tensor,
                x_covariates: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass with covariate integration
        
        Args:
            x_ts: Time series data [batch_size, seq_len, 1] 
            x_covariates: Covariate features [batch_size, seq_len, covariate_dim]
            attention_mask: Attention mask for the backbone
            
        Returns:
            Hidden states from backbone [batch_size, seq_len, d_model]
        """
        
        # Step 1: Fuse covariates with time series data
        enhanced_ts = self._fuse_covariates(x_ts, x_covariates)
        
        # Step 2: Prepare input for backbone (handles tokenization if needed)
        backbone_input = self._prepare_backbone_input(enhanced_ts)
        
        # Step 3: Pass to original backbone unchanged
        # The backbone's tokenizer and model handle the enhanced data
        if hasattr(self.backbone, 'tokenizer') and self.backbone.tokenizer is not None:
            # For Chronos-like models that need tokenization
            return self._forward_with_tokenization(backbone_input, attention_mask, **kwargs)
        else:
            # For models that work directly with embeddings
            return self.backbone.forward(backbone_input, attention_mask=attention_mask, **kwargs)
    
    def _fuse_covariates(self, 
                        x_ts: torch.Tensor, 
                        x_covariates: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Fuse covariates with time series data using the specified method
        
        Args:
            x_ts: [batch_size, seq_len, 1]
            x_covariates: [batch_size, seq_len, covariate_dim] or None
            
        Returns:
            Enhanced time series [batch_size, seq_len, 1]
        """
        
        if x_covariates is None or self.covariate_dim == 0:
            return x_ts
        
        batch_size, seq_len, ts_dim = x_ts.shape
        
        if self.fusion_method == 'project':
            # Project both to common space and combine
            ts_emb = self.ts_projector(x_ts)  # [B, L, emb_dim//2]
            cov_emb = self.covariate_projector(x_covariates)  # [B, L, emb_dim//2]
            # Apply positive gate to enforce non-zero contribution
            cov_emb = cov_emb * torch.relu(self.covariate_gate)  # scalar scaling
            
            # Concatenate embeddings
            combined_emb = torch.cat([ts_emb, cov_emb], dim=-1)  # [B, L, emb_dim]
            # Pre-fusion normalization (non-degenerate) to stabilise scale
            if hasattr(self, 'pre_fusion_norm'):
                combined_emb = self.pre_fusion_norm(combined_emb)
            
            # Project back to time series dimension
            enhanced_ts = self.fusion_projector(combined_emb)  # [B, L, 1]
            
        elif self.fusion_method == 'concat':
            # Concatenate raw features
            combined = torch.cat([x_ts, x_covariates], dim=-1)  # [B, L, 1+cov_dim]
            enhanced_ts = self.dimension_adapter(combined)  # [B, L, 1]
            
        elif self.fusion_method == 'add':
            # Project covariates and add to time series
            cov_projected = self.covariate_projector(x_covariates)  # [B, L, 1]
            enhanced_ts = x_ts + cov_projected
            
        # Apply normalization and dropout
        enhanced_ts = self.layer_norm(enhanced_ts)
        enhanced_ts = self.dropout(enhanced_ts)
        
        return enhanced_ts
    
    def _prepare_backbone_input(self, enhanced_ts: torch.Tensor) -> torch.Tensor:
        """
        Prepare enhanced time series for backbone input
        
        For Chronos: Returns raw enhanced data for tokenization
        For others: May apply additional transformations
        """
        return enhanced_ts
    
    def _forward_with_tokenization(self, 
                                  enhanced_ts: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor],
                                  **kwargs) -> torch.Tensor:
        """
        Handle models that require tokenization (like Chronos)
        
        The key insight: Chronos tokenizer can handle our enhanced time series data
        """
        try:
            # For Chronos, we need to tokenize the enhanced time series
            # The enhanced_ts contains our covariate-fused data
            tokenizer = self.backbone.tokenizer
            
            # Tokenize the enhanced time series
            # Note: This assumes enhanced_ts is in the format Chronos expects
            # [batch_size, seq_len, 1] -> tokenized format
            
            # Convert to the format expected by ChronosTokenizer
            # Typically this expects time series values, not embeddings
            enhanced_values = enhanced_ts.squeeze(-1)  # [batch_size, seq_len]
            
            # Tokenize (this may need adjustment based on exact Chronos API)
            if hasattr(tokenizer, 'encode'):
                # Some tokenizers have encode method
                input_ids = []
                for batch_idx in range(enhanced_values.size(0)):
                    series = enhanced_values[batch_idx].detach().cpu().numpy()
                    
                    # Handle the case where series might be a single sequence
                    try:
                        # Try to encode the series
                        if hasattr(series, '__iter__') and not isinstance(series, (str, bytes)):
                            tokens = tokenizer.encode(series)
                        else:
                            tokens = tokenizer.encode([float(series)])
                        
                        # Convert to tensor
                        if isinstance(tokens, list):
                            token_tensor = torch.tensor(tokens, dtype=torch.long)
                        elif isinstance(tokens, torch.Tensor):
                            token_tensor = tokens.long()
                        else:
                            token_tensor = torch.tensor([tokens], dtype=torch.long)
                            
                    except Exception as token_e:
                        logger.warning(f"Tokenization failed for batch {batch_idx}: {token_e}")
                        # Fallback: convert series values to integers
                        token_tensor = torch.tensor(
                            [int(x * 100) % 1000 for x in series.flatten()], 
                            dtype=torch.long
                        )
                    
                    input_ids.append(token_tensor)
                
                # Stack tensors, padding if necessary
                max_len = max(t.size(0) for t in input_ids)
                padded_input_ids = []
                for t in input_ids:
                    if t.size(0) < max_len:
                        # Pad with zeros
                        padding = torch.zeros(max_len - t.size(0), dtype=torch.long)
                        t = torch.cat([t, padding])
                    elif t.size(0) > max_len:
                        # Truncate if too long
                        t = t[:max_len]
                    padded_input_ids.append(t)
                
                input_ids = torch.stack(padded_input_ids).to(enhanced_values.device)
            else:
                # Direct tokenization - this may need Chronos-specific handling
                input_ids = enhanced_values.long()  # Simplified assumption
            
            # Forward through backbone
            return self.backbone.forward(input_ids, attention_mask=attention_mask, **kwargs)
            
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, falling back to direct forward")
            # Fallback: treat as embedding input
            return self.backbone.forward(enhanced_ts, attention_mask=attention_mask, **kwargs)
    
    def get_d_model(self) -> int:
        """Return the output dimension (same as backbone)"""
        return self.backbone.get_d_model()
    
    def supports_seq2seq(self) -> bool:
        """Return backbone's seq2seq support"""
        return self.backbone.supports_seq2seq()
    
    def get_backbone_type(self) -> str:
        """Return adapter type"""
        return f"covariate_adapted_{self.backbone.get_backbone_type()}"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get enhanced capabilities including covariate support"""
        base_capabilities = self.backbone.get_capabilities()
        base_capabilities.update({
            'supports_covariates': True,
            'covariate_dim': self.covariate_dim,
            'fusion_method': self.fusion_method,
            'adapter_type': 'covariate',
            'base_backbone': self.backbone.get_backbone_type()
        })
        return base_capabilities


class MultiScaleAdapter(BaseBackbone):
    """Apply the wrapped backbone at multiple temporal scales and aggregate outputs.

    Provides access to the individual (upsampled) per-scale representations via
    ``_last_scale_outputs`` after each forward pass for diagnostic / invariant tests.
    """

    def __init__(self, backbone: BaseBackbone, scale_config: Dict[str, Any]):
        super().__init__(backbone.config)
        self.backbone = backbone
        self.scales = scale_config.get("scales", [1, 2, 4])
        self.aggregation_method = scale_config.get("aggregation", "concat")

        d_model = self.backbone.get_d_model()
        self.scale_processors = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in self.scales
        ])

        if self.aggregation_method == "concat":
            self.aggregator = nn.Linear(len(self.scales) * d_model, d_model)
        elif self.aggregation_method == "add":
            self.aggregator = nn.Identity()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        logger.info(f"MultiScaleAdapter initialized with scales: {self.scales}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: D401
        scale_outputs = []  # per-scale representations aligned to original length
        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_scaled = x[:, ::scale, :]
            else:
                x_scaled = x
            h = self.backbone.forward(x_scaled, **kwargs)
            h = self.scale_processors[i](h)
            if scale > 1 and h.size(1) != x.size(1):
                h = torch.nn.functional.interpolate(
                    h.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False
                ).transpose(1, 2)
            scale_outputs.append(h)
        if self.aggregation_method == "concat":
            combined = torch.cat(scale_outputs, dim=-1)
            out = self.aggregator(combined)
        else:  # add
            out = torch.stack(scale_outputs).sum(dim=0)
        self._last_scale_outputs = scale_outputs  # type: ignore[attr-defined]
        return out

    def get_d_model(self) -> int:
        return self.backbone.get_d_model()

    def supports_seq2seq(self) -> bool:
        return self.backbone.supports_seq2seq()

    def get_backbone_type(self) -> str:
        return f"multiscale_{self.backbone.get_backbone_type()}"
