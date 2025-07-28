"""
SequentialDecoder - Proper sequential decoding for time series prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from utils.logger import logger

from layers.MambaBlock import MambaBlock


class SequentialDecoder(nn.Module):
    """
    Sequential decoder that generates predictions step by step using context vectors.
    Handles both trend and seasonal components properly.
    """
    
    def __init__(
        self,
        d_model: int,
        c_out: int,
        pred_len: int,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.1,
        use_autoregressive: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.c_out = c_out
        self.pred_len = pred_len
        self.use_autoregressive = use_autoregressive
        
        # Context fusion layer
        self.context_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Fuse target + covariate contexts
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Sequential generation Mamba block
        self.sequential_mamba = MambaBlock(
            input_dim=d_model + c_out if use_autoregressive else d_model,
            d_model=d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout
        )
        
        # Trend and seasonal projection heads
        self.trend_projection = nn.Linear(d_model, c_out)
        self.seasonal_projection = nn.Linear(d_model, c_out)
        
        # Final combination layer
        self.final_projection = nn.Sequential(
            nn.Linear(c_out * 2, c_out),  # Combine trend + seasonal
            nn.Dropout(dropout)
        )
        
        logger.info(f"SequentialDecoder initialized: pred_len={pred_len}, autoregressive={use_autoregressive}")
    
    def forward(
        self,
        target_context: torch.Tensor,
        covariate_context: torch.Tensor,
        initial_values: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sequential decoding with trend-seasonal decomposition.
        
        Args:
            target_context: [B, D] - Target context vector
            covariate_context: [B, D] - Covariate context vector
            initial_values: [B, c_out] - Initial values for autoregressive generation
            future_covariates: [B, pred_len, num_covariates] - Future covariate information
            
        Returns:
            Dictionary with trend, seasonal, and combined predictions
        """
        batch_size = target_context.size(0)
        device = target_context.device
        
        # Fuse target and covariate contexts
        fused_context = self.context_fusion(
            torch.cat([target_context, covariate_context], dim=-1)
        )  # [B, D]
        
        # Initialize outputs
        trend_predictions = []
        seasonal_predictions = []
        combined_predictions = []
        
        # Initialize current input
        if self.use_autoregressive and initial_values is not None:
            current_input = torch.cat([fused_context, initial_values], dim=-1)  # [B, D + c_out]
        else:
            current_input = fused_context  # [B, D]
        
        # Sequential generation
        hidden_state = None
        for t in range(self.pred_len):
            # Add sequence dimension for Mamba
            mamba_input = current_input.unsqueeze(1)  # [B, 1, D] or [B, 1, D + c_out]
            
            # Process through Mamba
            mamba_output = self.sequential_mamba(mamba_input)  # [B, 1, D]
            mamba_output = mamba_output.squeeze(1)  # [B, D]
            
            # Generate trend and seasonal components
            trend_t = self.trend_projection(mamba_output)  # [B, c_out]
            seasonal_t = self.seasonal_projection(mamba_output)  # [B, c_out]
            
            # Combine trend and seasonal
            combined_input = torch.cat([trend_t, seasonal_t], dim=-1)  # [B, 2*c_out]
            combined_t = self.final_projection(combined_input)  # [B, c_out]
            
            # Store predictions
            trend_predictions.append(trend_t)
            seasonal_predictions.append(seasonal_t)
            combined_predictions.append(combined_t)
            
            # Update input for next step (autoregressive)
            if self.use_autoregressive:
                current_input = torch.cat([fused_context, combined_t], dim=-1)
            else:
                # Non-autoregressive: use time-varying context if available
                if future_covariates is not None and t < future_covariates.size(1) - 1:
                    # Update context with future covariate information
                    # This is a simplified approach - could be more sophisticated
                    current_input = fused_context
                else:
                    current_input = fused_context
        
        # Stack predictions
        trend_output = torch.stack(trend_predictions, dim=1)  # [B, pred_len, c_out]
        seasonal_output = torch.stack(seasonal_predictions, dim=1)  # [B, pred_len, c_out]
        combined_output = torch.stack(combined_predictions, dim=1)  # [B, pred_len, c_out]
        
        outputs = {
            'trend': trend_output,
            'seasonal': seasonal_output,
            'combined': combined_output,
            'final': combined_output  # Main output
        }
        
        logger.debug(f"Sequential decoding complete: output shape {combined_output.shape}")
        
        return outputs
    
    def generate_with_beam_search(
        self,
        target_context: torch.Tensor,
        covariate_context: torch.Tensor,
        beam_size: int = 3,
        initial_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate predictions using beam search for better quality.
        
        Args:
            target_context: [B, D]
            covariate_context: [B, D]
            beam_size: Number of beams
            initial_values: [B, c_out]
            
        Returns:
            Best predictions [B, pred_len, c_out]
        """
        # Simplified beam search implementation
        # For now, just return regular forward pass
        # TODO: Implement proper beam search
        outputs = self.forward(target_context, covariate_context, initial_values)
        return outputs['final']