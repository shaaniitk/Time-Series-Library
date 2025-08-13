"""
Simplified Modular Autoformer - Practical Implementation

This demonstrates a simplified approach to the modular framework that prioritizes
usability and maintainability over architectural complexity.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class AutoformerVariant(str, Enum):
    """Simplified variant enumeration"""
    STANDARD = "standard"
    ENHANCED = "enhanced" 
    BAYESIAN = "bayesian"
    HIERARCHICAL = "hierarchical"
    QUANTILE = "quantile"

@dataclass
class SimpleAutoformerConfig:
    """Simplified configuration with sensible defaults"""
    # Basic parameters
    seq_len: int = 96
    pred_len: int = 24
    label_len: int = 48
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    d_model: int = 512
    
    # Model variant
    variant: AutoformerVariant = AutoformerVariant.STANDARD
    
    # Component options (simplified)
    use_adaptive_attention: bool = False
    use_learnable_decomp: bool = False
    use_bayesian_layers: bool = False
    quantile_levels: Optional[list] = None
    
    # Standard hyperparameters
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    moving_avg: int = 25

class SimpleComponent(nn.Module):
    """Base class for simplified components"""
    
    def __init__(self, config: SimpleAutoformerConfig):
        super().__init__()
        self.config = config

class SimpleAttention(SimpleComponent):
    """Unified attention component with optional enhancements"""
    
    def __init__(self, config: SimpleAutoformerConfig):
        super().__init__(config)
        
        # Import the appropriate attention based on config
        if config.use_adaptive_attention:
            from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation, AdaptiveAutoCorrelationLayer
            self.attention = AdaptiveAutoCorrelationLayer(
                AdaptiveAutoCorrelation(False, 1, attention_dropout=config.dropout),
                config.d_model, config.n_heads
            )
        else:
            from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
            self.attention = AutoCorrelationLayer(
                AutoCorrelation(False, 1, attention_dropout=config.dropout),
                config.d_model, config.n_heads
            )
    
    def forward(self, queries, keys, values, attn_mask=None):
        return self.attention(queries, keys, values, attn_mask)

class SimpleDecomposition(SimpleComponent):
    """Unified decomposition component"""
    
    def __init__(self, config: SimpleAutoformerConfig):
        super().__init__(config)
        
        if config.use_learnable_decomp:
            # Learnable decomposition
            self.decomp = nn.Conv1d(
                config.d_model, config.d_model, 
                config.moving_avg, padding=config.moving_avg//2
            )
            self.learnable = True
        else:
            # Standard moving average
            from layers.Autoformer_EncDec import series_decomp
            self.decomp = series_decomp(config.moving_avg)
            self.learnable = False
    
    def forward(self, x):
        if self.learnable:
            # Simple learnable decomposition
            x_permuted = x.permute(0, 2, 1)
            trend = self.decomp(x_permuted).permute(0, 2, 1)
            seasonal = x - trend
            return seasonal, trend
        else:
            return self.decomp(x)

class SimpleEncoder(SimpleComponent):
    """Simplified encoder with optional enhancements"""
    
    def __init__(self, config: SimpleAutoformerConfig):
        super().__init__(config)
        
        # Use existing encoder implementations
        from layers.Autoformer_EncDec import Encoder, EncoderLayer
        
        # Create attention and decomposition components
        attention_comp = SimpleAttention(config)
        decomp_comp = SimpleDecomposition(config)
        
        # Build encoder layers
        self.encoder = Encoder([
            EncoderLayer(
                attention_comp.attention,
                config.d_model,
                config.d_ff,
                moving_avg=config.moving_avg,
                dropout=config.dropout,
                activation=config.activation
            ) for _ in range(config.e_layers)
        ])
    
    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)

class SimpleDecoder(SimpleComponent):
    """Simplified decoder"""
    
    def __init__(self, config: SimpleAutoformerConfig):
        super().__init__(config)
        
        from layers.Autoformer_EncDec import Decoder, DecoderLayer
        
        # Create attention and decomposition components
        attention_comp = SimpleAttention(config)
        decomp_comp = SimpleDecomposition(config)
        
        # Build decoder layers
        self.decoder = Decoder([
            DecoderLayer(
                attention_comp.attention,  # self attention
                attention_comp.attention,  # cross attention
                config.d_model,
                config.c_out,
                config.d_ff,
                moving_avg=config.moving_avg,
                dropout=config.dropout,
                activation=config.activation
            ) for _ in range(config.d_layers)
        ], projection=nn.Linear(config.d_model, config.c_out))
    
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)

class SimplifiedModularAutoformer(nn.Module):
    """
    Simplified modular Autoformer that achieves the same goals with less complexity
    """
    
    def __init__(self, config: SimpleAutoformerConfig):
        super().__init__()
        self.config = config
        
        # Apply variant-specific modifications
        self._apply_variant_config()
        
        # Initialize components
        self._init_components()
    
    def _apply_variant_config(self):
        """Apply variant-specific configuration changes"""
        if self.config.variant == AutoformerVariant.ENHANCED:
            self.config.use_adaptive_attention = True
            self.config.use_learnable_decomp = True
        
        elif self.config.variant == AutoformerVariant.BAYESIAN:
            self.config.use_bayesian_layers = True
            self.config.use_adaptive_attention = True
        
        elif self.config.variant == AutoformerVariant.HIERARCHICAL:
            self.config.use_adaptive_attention = True
            # Could add hierarchical-specific settings
        
        elif self.config.variant == AutoformerVariant.QUANTILE:
            if self.config.quantile_levels is None:
                self.config.quantile_levels = [0.1, 0.5, 0.9]
            # Adjust output dimension for quantiles
            self.config.c_out = self.config.c_out * len(self.config.quantile_levels)
    
    def _init_components(self):
        """Initialize model components"""
        from layers.Embed import DataEmbedding_wo_pos
        from layers.Autoformer_EncDec import series_decomp
        
        # Embeddings
        self.enc_embedding = DataEmbedding_wo_pos(
            self.config.enc_in, self.config.d_model, 
            'timeF', 'h', self.config.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            self.config.dec_in, self.config.d_model,
            'timeF', 'h', self.config.dropout
        )
        
        # Core components
        self.decomp = SimpleDecomposition(self.config)
        self.encoder = SimpleEncoder(self.config)
        self.decoder = SimpleDecoder(self.config)
        
        # Apply Bayesian conversion if needed
        if self.config.use_bayesian_layers:
            self._apply_bayesian_conversion()
    
    def _apply_bayesian_conversion(self):
        """Convert specified layers to Bayesian variants"""
        try:
            from layers.BayesianLayers import make_bayesian
            
            # Convert decoder projection to Bayesian
            if hasattr(self.decoder.decoder, 'projection'):
                self.decoder.decoder.projection = make_bayesian(
                    self.decoder.decoder.projection
                )
        except ImportError:
            print("Warning: Bayesian layers not available")
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Simplified forward pass"""
        
        # Decomposition initialization
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.config.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.config.pred_len, x_dec.shape[2]], device=x_enc.device)
        
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Prepare decoder inputs
        trend_init = torch.cat([trend_init[:, -self.config.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.config.label_len:, :], zeros], dim=1)
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        
        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        
        # Final output
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.config.pred_len:, :]

# Factory functions for easy model creation
def create_standard_autoformer(**kwargs) -> SimplifiedModularAutoformer:
    """Create standard Autoformer"""
    config = SimpleAutoformerConfig(variant=AutoformerVariant.STANDARD, **kwargs)
    return SimplifiedModularAutoformer(config)

def create_enhanced_autoformer(**kwargs) -> SimplifiedModularAutoformer:
    """Create enhanced Autoformer with adaptive features"""
    config = SimpleAutoformerConfig(variant=AutoformerVariant.ENHANCED, **kwargs)
    return SimplifiedModularAutoformer(config)

def create_bayesian_autoformer(**kwargs) -> SimplifiedModularAutoformer:
    """Create Bayesian Autoformer with uncertainty quantification"""
    config = SimpleAutoformerConfig(variant=AutoformerVariant.BAYESIAN, **kwargs)
    return SimplifiedModularAutoformer(config)

def create_quantile_autoformer(quantile_levels=None, **kwargs) -> SimplifiedModularAutoformer:
    """Create quantile Autoformer for probabilistic forecasting"""
    config = SimpleAutoformerConfig(
        variant=AutoformerVariant.QUANTILE, 
        quantile_levels=quantile_levels or [0.1, 0.5, 0.9],
        **kwargs
    )
    return SimplifiedModularAutoformer(config)

# Usage examples
if __name__ == "__main__":
    # Simple usage - much cleaner than the complex modular approach
    
    # Standard model
    model = create_standard_autoformer(
        seq_len=96, pred_len=24, enc_in=7, c_out=7
    )
    
    # Enhanced model with adaptive features
    enhanced_model = create_enhanced_autoformer(
        seq_len=96, pred_len=24, enc_in=7, c_out=7
    )
    
    # Bayesian model for uncertainty
    bayesian_model = create_bayesian_autoformer(
        seq_len=96, pred_len=24, enc_in=7, c_out=7
    )
    
    # Quantile model for probabilistic forecasting
    quantile_model = create_quantile_autoformer(
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        seq_len=96, pred_len=24, enc_in=7, c_out=1  # Note: c_out will be multiplied by num_quantiles
    )
    
    print("✅ All models created successfully with simplified interface")