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
                
        # Sequential generation Mamba block
        self.sequential_mamba = MambaBlock(
            input_dim=d_model + c_out if use_autoregressive else d_model, # Input is context + previous output
            d_model=d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout
        )

        # Final projection head
        self.projection = nn.Linear(d_model, c_out)
        
        logger.info(f"SequentialDecoder initialized: pred_len={pred_len}, autoregressive={use_autoregressive}")
    
    def forward(
        self,
        context: torch.Tensor,
        initial_values: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        teacher_forcing_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sequential decoding with trend-seasonal decomposition.
        
        Args:
            context: [B, D] - Fused context vector
            initial_values: [B, c_out] - Initial values for autoregressive generation
            future_covariates: [B, pred_len, num_covariates] - Future covariate information
            teacher_forcing_targets: [B, pred_len, c_out] - Ground-truth targets for teacher forcing during training.
            
        Returns:
            Dictionary with trend, seasonal, and combined predictions
        """
        batch_size = context.size(0)
        device = context.device
        
        # Initialize outputs
        predictions = []
        
        # Initialize current input
        if self.use_autoregressive and initial_values is not None:
            current_input = torch.cat([context, initial_values], dim=-1)  # [B, D + c_out]
        else:
            current_input = context  # [B, D]
        
        # Sequential generation
        for t in range(self.pred_len):
            # Add sequence dimension for Mamba
            mamba_input = current_input.unsqueeze(1)  # [B, 1, D] or [B, 1, D + c_out]
            
            # Process through Mamba
            mamba_output = self.sequential_mamba(mamba_input)  # [B, 1, D]
            mamba_output = mamba_output.squeeze(1)  # [B, D]
            
            # Generate prediction for this step
            pred_t = self.projection(mamba_output) # [B, c_out]
            
            # Store predictions
            predictions.append(pred_t)
            
            # Update input for next step (autoregressive)
            if self.use_autoregressive:
                # Use teacher forcing if targets are provided (i.e., during training)
                if teacher_forcing_targets is not None and t < self.pred_len - 1:
                    current_input = torch.cat([context, teacher_forcing_targets[:, t, :]], dim=-1)
                else: # Use its own prediction for the next step (inference or last step)
                    current_input = torch.cat([context, pred_t], dim=-1)
            else:
                # Non-autoregressive: context remains the same
                current_input = context
        
        # Stack predictions
        final_output = torch.stack(predictions, dim=1)  # [B, pred_len, c_out]
        
        outputs = {
            'final': final_output  # Main output
        }
        
        logger.debug(f"Sequential decoding complete: output shape {final_output.shape}")
        
        return outputs
    
    def generate_with_beam_search(
        self,
        context: torch.Tensor,
        beam_size: int = 5,
        initial_values: Optional[torch.Tensor] = None,
        temperature: float = 0.5
    ) -> torch.Tensor:
        """
        Generate predictions using beam search for better quality.
        
        Since this is a regression model, beam search is adapted by sampling
        from a Gaussian distribution centered on the model's output at each step.
        This allows exploring multiple candidate sequences.
        
        Args:
            context: [B, D] - Fused context vector
            beam_size: Number of beams
            initial_values: [B, c_out]
            temperature: Standard deviation for sampling. Higher values lead to more diversity.
            
        Returns:
            Best predictions [B, pred_len, c_out]
        """
        B, D = context.shape
        k = beam_size
        device = context.device

        if initial_values is None:
            initial_values = torch.zeros(B, self.c_out, device=device)

        # Expand context for all beams: [B, D] -> [B*k, D]
        expanded_context = context.unsqueeze(1).expand(-1, k, -1).reshape(B * k, D)

        # Initialize sequences [B, k, pred_len, c_out] and scores [B, k]
        sequences = torch.zeros(B, k, self.pred_len, self.c_out, device=device)
        scores = torch.zeros(B, k, device=device)
        
        # First input is the initial_values, expanded for each beam
        last_preds = initial_values.unsqueeze(1).expand(-1, k, -1)  # [B, k, c_out]

        for t in range(self.pred_len):
            # Prepare input for all beams at once: [B*k, D+c_out]
            current_input = torch.cat([expanded_context, last_preds.reshape(B * k, self.c_out)], dim=-1)
            
            # Get model output (means of distributions): [B*k, c_out]
            mamba_output = self.sequential_mamba(current_input.unsqueeze(1)).squeeze(1)
            means = self.projection(mamba_output)
            
            # Reshape means to separate batches and beams: [B, k, c_out]
            means = means.view(B, k, self.c_out)
            
            # Create a normal distribution centered at the means
            dist = torch.distributions.Normal(means, temperature)
            
            # Sample k candidates for each of the k beams: [B, k, k, c_out]
            candidates = dist.sample((k,)).permute(1, 2, 0, 3)
            
            # Calculate log probabilities and add to existing scores
            log_probs = dist.log_prob(candidates).sum(dim=-1)  # [B, k, k]
            combined_scores = scores.unsqueeze(-1) + log_probs  # [B, k, k]

            # Flatten scores to find the top k across all candidates
            flat_scores, top_indices = torch.topk(combined_scores.view(B, -1), k, dim=-1) # [B, k]
            
            # Determine the source beam and candidate index for each of the top k scores
            beam_indices = top_indices // k
            candidate_indices = top_indices % k
            
            # Gather the best candidates and update sequences
            batch_indices = torch.arange(B, device=device).unsqueeze(-1)
            sequences = sequences[batch_indices, beam_indices]
            last_preds = candidates[batch_indices, beam_indices, candidate_indices]
            sequences[:, :, t, :] = last_preds
            scores = flat_scores

        # Select the best sequence (highest score) for each item in the batch
        best_beam_indices = torch.argmax(scores, dim=1)
        best_sequences = sequences[torch.arange(B, device=device), best_beam_indices]
        
        return best_sequences