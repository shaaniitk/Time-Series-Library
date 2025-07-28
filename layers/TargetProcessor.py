"""
TargetProcessor - Handles wavelet decomposition → mamba → attention pipeline for target variables
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from utils.logger import logger

from layers.MambaBlock import TargetMambaBlock
from layers.modular.decomposition.wavelet_decomposition import WaveletDecomposition
from layers.modular.attention.enhanced_autocorrelation import EnhancedAutoCorrelation


class TargetProcessor(nn.Module):
    """
    Processes target variables through: Wavelet Decomposition → Mamba → Multi-Head Attention
    Outputs a single context vector representing target information.
    """
    
    def __init__(
        self,
        num_targets: int,
        seq_len: int,
        d_model: int,
        wavelet_type: str = 'db4',
        wavelet_levels: int = 3,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
        use_trend_decomposition: bool = False
    ):
        super(TargetProcessor, self).__init__()
        
        self.num_targets = num_targets
        self.seq_len = seq_len
        self.d_model = d_model
        self.wavelet_levels = wavelet_levels
        self.attention_heads = attention_heads
        
        # Input projection from num_targets to d_model
        self.target_projection = nn.Linear(num_targets, d_model)
        
        # Wavelet decomposition component
        try:
            from layers.modular.decomposition.wavelet_decomposition import WaveletDecomposition
            self.wavelet_decomp = WaveletDecomposition(
                seq_len=seq_len,
                d_model=d_model,
                wavelet_type=wavelet_type,
                levels=wavelet_levels
            )
            logger.info(f"Using modular WaveletDecomposition with {wavelet_levels} levels")
        except ImportError:
            # Fallback to basic wavelet if modular not available
            logger.warning("Modular WaveletDecomposition not found, using basic implementation")
            self.wavelet_decomp = self._create_basic_wavelet_decomp(wavelet_type, wavelet_levels)
        
        # Mamba block for each decomposition level
        self.mamba_blocks = nn.ModuleList([
            TargetMambaBlock(
                input_dim=d_model,  # Now using projected inputs
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
                use_trend_decomposition=use_trend_decomposition
            ) for _ in range(wavelet_levels + 1)  # +1 for approximation component
        ])
        
        # Multi-head attention for combining decomposed components
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Context aggregation - will be created dynamically based on actual input size
        self.context_aggregator = None
        self.expected_aggregator_input_size = d_model * (wavelet_levels + 1)
        
        # Output projection to create final context vector
        self.context_projection = nn.Linear(d_model, d_model)
        
        logger.info(f"TargetProcessor initialized: num_targets={num_targets}, "
                   f"wavelet_levels={wavelet_levels}, d_model={d_model}")
    
    def _create_basic_wavelet_decomp(self, wavelet_type: str, levels: int):
        """Create basic wavelet decomposition if modular version not available."""
        class BasicWaveletDecomp(nn.Module):
            def __init__(self, wavelet_type, levels):
                super().__init__()
                self.wavelet_type = wavelet_type
                self.levels = levels
                
            def forward(self, x):
                # Improved fallback: frequency-based splitting using FFT
                batch_size, seq_len, features = x.shape
                
                # Apply FFT to get frequency components
                x_fft = torch.fft.fft(x, dim=1)
                
                components = []
                freq_bands = seq_len // (self.levels + 1)
                
                for i in range(self.levels + 1):
                    # Create frequency mask for this level
                    mask = torch.zeros_like(x_fft)
                    start_freq = i * freq_bands
                    end_freq = min((i + 1) * freq_bands, seq_len // 2)
                    mask[:, start_freq:end_freq, :] = 1
                    mask[:, -end_freq:-start_freq if start_freq > 0 else seq_len, :] = 1
                    
                    # Apply mask and inverse FFT
                    filtered_fft = x_fft * mask
                    component = torch.fft.ifft(filtered_fft, dim=1).real
                    components.append(component)
                
                return components
        
        return BasicWaveletDecomp(wavelet_type, levels)
    
    def forward(
        self, 
        targets: torch.Tensor, 
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process target variables through wavelet → mamba → attention pipeline.
        
        Args:
            targets: Target variables tensor [batch_size, seq_len, num_targets]
            target_mask: Optional mask for targets
            
        Returns:
            context_vector: Single context vector [batch_size, d_model]
        """
        batch_size, seq_len, num_features = targets.shape
        
        logger.debug(f"TargetProcessor forward: input shape {targets.shape}")
        
        # Step 1: Project targets to d_model space
        projected_targets = self.target_projection(targets)
        logger.debug(f"Projected targets shape: {projected_targets.shape}")
        
        # Step 2: Wavelet decomposition
        try:
            wavelet_components = self.wavelet_decomp(projected_targets)
            logger.debug(f"Wavelet decomposition produced {len(wavelet_components)} components")
        except Exception as e:
            logger.error(f"Wavelet decomposition failed: {e}")
            # Fallback: use projected signal
            wavelet_components = [projected_targets]
        
        # Step 2: Process each component through Mamba
        mamba_outputs = []
        for i, component in enumerate(wavelet_components):
            if i < len(self.mamba_blocks):
                try:
                    mamba_out = self.mamba_blocks[i](component, target_mask)
                    mamba_outputs.append(mamba_out)
                    logger.debug(f"Mamba block {i} output shape: {mamba_out.shape}")
                except Exception as e:
                    logger.error(f"Mamba block {i} failed: {e}")
                    continue
        
        if not mamba_outputs:
            logger.error("No valid Mamba outputs, using fallback")
            # Fallback: process projected targets through first Mamba block
            mamba_outputs = [self.mamba_blocks[0](projected_targets, target_mask)]
        
        # Step 3: Apply multi-head attention across components
        # Concatenate all components along sequence dimension for attention
        try:
            # Ensure all outputs have same sequence length by padding/truncating
            max_len = max(out.size(1) for out in mamba_outputs)
            padded_outputs = []
            
            for out in mamba_outputs:
                if out.size(1) < max_len:
                    # Pad shorter sequences
                    padding = torch.zeros(batch_size, max_len - out.size(1), self.d_model, 
                                        device=out.device, dtype=out.dtype)
                    padded_out = torch.cat([out, padding], dim=1)
                else:
                    # Truncate longer sequences
                    padded_out = out[:, :max_len, :]
                padded_outputs.append(padded_out)
            
            # Stack components for attention
            stacked_components = torch.stack(padded_outputs, dim=2)  # [B, L, num_components, D]
            B, L, C, D = stacked_components.shape
            
            # Reshape for attention: [B, L*C, D]
            attention_input = stacked_components.view(B, L * C, D)
            
            # Apply multi-head attention
            attended_output, attention_weights = self.multi_head_attention(
                attention_input, attention_input, attention_input
            )
            
            logger.debug(f"Multi-head attention output shape: {attended_output.shape}")
            
        except Exception as e:
            logger.error(f"Multi-head attention failed: {e}")
            # Fallback: simple concatenation
            attended_output = torch.cat(mamba_outputs, dim=-1)
        
        # Step 4: Aggregate to context vector
        try:
            if attended_output.dim() == 3:
                # Global average pooling across sequence dimension
                pooled_output = attended_output.mean(dim=1)  # [B, D]
            else:
                pooled_output = attended_output
            
            # Apply context aggregation with dynamic sizing
            if pooled_output.size(-1) != self.d_model:
                # Create context aggregator if not exists or size changed
                if self.context_aggregator is None or self.context_aggregator[0].in_features != pooled_output.size(-1):
                    self.context_aggregator = nn.Sequential(
                        nn.Linear(pooled_output.size(-1), self.d_model * 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.d_model * 2, self.d_model),
                        nn.LayerNorm(self.d_model)
                    ).to(pooled_output.device)
                context_vector = self.context_aggregator(pooled_output)
            else:
                context_vector = pooled_output
            
            # Final projection
            context_vector = self.context_projection(context_vector)
            
            logger.debug(f"Final target context vector shape: {context_vector.shape}")
            
        except Exception as e:
            logger.error(f"Context aggregation failed: {e}")
            # Fallback: simple mean pooling
            context_vector = torch.mean(torch.stack(mamba_outputs), dim=0).mean(dim=1)
            context_vector = self.context_projection(context_vector)
        
        return context_vector
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the last computed attention weights for visualization."""
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        return None